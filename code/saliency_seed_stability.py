import csv
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy.ndimage import zoom
from scipy.stats import pearsonr, spearmanr, combine_pvalues
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from data_utils import DEVICE, ASDVolumeDataset, load_and_preprocess, adapt_batchnorm_to_domain
from medicalnet import build_backbone, load_pretrained_backbone
from average_saliency_by_region import REGION_CODES, FREESURFER_DIR

CACHE_PATH = "/path/to/cached_features_fullres.pt"
CKPT_PATH = "/path/to/resnet_50_23dataset.pth"
MODEL_DEPTH = 50
ADABN_BATCH_SIZE = 1

ALL_HEADS_DIR = "/path/to/all_runs"
N_SAVED_RUNS = 1000

OUT_DIR = "/path/to/saliency_seed_stability_ig"
EXAMPLES_DIR = os.path.join(OUT_DIR, "example_maps")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

PAIR_CSV = os.path.join(OUT_DIR, "seed_pair_correlations_ig.csv")
SUBJECT_SUMMARY_CSV = os.path.join(OUT_DIR, "seed_stability_per_subject_ig.csv")
POPULATION_CSV = os.path.join(OUT_DIR, "seed_stability_population_level_ig.csv")

N_SEED_RUNS = 5
SAMPLE_SEED = 424242
N_SUBJECTS_PER_FOLD = 4
SUBJECT_SAMPLE_SEED = 20260817
SAVE_EXAMPLE_MAPS_PER_FOLD = 2
K_TOP_REGIONS = 20

TARGET = "high_severity_logit"
BASELINE = "zero"
N_STEPS = 50
INTERNAL_BATCH_SIZE = 1


pool = nn.AdaptiveAvgPool3d(1)


class MedNetWithHead(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = pool(self.backbone(x)).flatten(1)
        return self.head(feat)


def build_frozen_backbone():
    backbone = build_backbone(MODEL_DEPTH)
    load_pretrained_backbone(backbone, CKPT_PATH, device=DEVICE, verbose=False)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def load_head_for_run(run_idx, fold):
    head = nn.Linear(2048, 1).to(DEVICE)
    state = torch.load(
        os.path.join(ALL_HEADS_DIR, f"run{run_idx}_fold{fold}.pt"), map_location=DEVICE
    )
    head.load_state_dict(state)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head


def realign_to_mask_space(sal):
    sal = np.flip(sal, axis=2)
    sal = np.transpose(sal, (0, 2, 1))
    return sal.T


def make_baseline(vol):
    if BASELINE == "zero":
        return torch.zeros_like(vol)
    raise ValueError(f"Unknown BASELINE: {BASELINE}")


def compute_ig(model, vol_np):
    vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    baseline = make_baseline(vol)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        vol, baselines=baseline, target=0,
        n_steps=N_STEPS, internal_batch_size=INTERNAL_BATCH_SIZE,
        return_convergence_delta=True,
    )

    with torch.no_grad():
        logit_input = model(vol).squeeze()
        logit_baseline = model(baseline).squeeze()

    attr = attributions.detach().cpu().numpy()[0, 0]
    prob = torch.sigmoid(logit_input).item()
    conv_delta = float(delta.abs().item())
    completeness_gap = float((logit_input - logit_baseline).abs().item())
    relative_delta = conv_delta / (completeness_gap + 1e-8)
    return attr, prob, conv_delta, relative_delta


def load_subject_region_masks(subject, shape):
    masks = {}
    for region_code in REGION_CODES:
        mask_path = os.path.join(FREESURFER_DIR, str(subject), f"{region_code}.mgz")
        if not os.path.exists(mask_path):
            continue
        m = nib.load(mask_path).get_fdata()
        if m.shape != shape:
            continue
        masks[region_code] = (m == 1.0)
    return masks


def foreground_mask_from_cache(masks, shape):
    fg = np.zeros(shape, dtype=bool)
    for m in masks.values():
        fg |= m
    return fg


def region_vector_from_cache(sal, masks):
    vec = np.full(len(REGION_CODES), np.nan, dtype=np.float64)
    for i, region_code in enumerate(REGION_CODES):
        m = masks.get(region_code)
        if m is not None and m.any():
            vec[i] = float(sal[m].mean())
    return vec


def safe_corr(a, b):
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    pear, _ = pearsonr(a, b)
    spear, _ = spearmanr(a, b)
    return float(pear), float(spear)


def region_pair_corr(vec_a, vec_b):
    valid = ~(np.isnan(vec_a) | np.isnan(vec_b))
    n_valid = int(valid.sum())
    if n_valid < 3:  # need df=n-2>=1 for a meaningful p-value
        return float("nan"), float("nan"), float("nan"), float("nan"), n_valid
    a, b = vec_a[valid], vec_b[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_valid
    pear_r, pear_p = pearsonr(a, b)
    spear_r, spear_p = spearmanr(a, b)
    return float(pear_r), float(pear_p), float(spear_r), float(spear_p), n_valid


DOWNSAMPLE_FACTOR = 0.5   # shift-null only -- validated to closely track full-resolution
N_SHIFTS = 200


def shift_permutation_pvalue(map_a, map_b, mask, seed, n_shifts=N_SHIFTS, downsample=DOWNSAMPLE_FACTOR):
    a_small = zoom(map_a, downsample, order=1)
    b_small = zoom(map_b, downsample, order=1)
    mask_small = zoom(mask.astype(np.float32), downsample, order=0) > 0.5

    a_vals = a_small[mask_small]
    if a_vals.size < 2 or np.std(a_vals) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    b_vals = b_small[mask_small]
    if np.std(b_vals) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs = float(abs(np.corrcoef(a_vals, b_vals)[0, 1]))

    rng = np.random.RandomState(seed)
    null_r = []
    for _ in range(n_shifts):
        shift = tuple(rng.randint(1, s) for s in b_small.shape)
        shifted = np.roll(b_small, shift=shift, axis=(0, 1, 2))
        b_v = shifted[mask_small]
        if np.std(b_v) == 0:
            continue
        null_r.append(abs(np.corrcoef(a_vals, b_v)[0, 1]))
    null_r = np.array(null_r)
    if len(null_r) == 0:
        return obs, float("nan"), float("nan"), float("nan")

    p = float((np.sum(null_r >= obs) + 1) / (len(null_r) + 1))
    return obs, p, float(null_r.mean()), float(null_r.std())


def topk_jaccard(vec_a, vec_b, k=K_TOP_REGIONS):
    valid = ~(np.isnan(vec_a) | np.isnan(vec_b))
    idx = np.where(valid)[0]
    if len(idx) < k:
        return float("nan")
    top_a = set(idx[np.argsort(-np.abs(vec_a[idx]))[:k]])
    top_b = set(idx[np.argsort(-np.abs(vec_b[idx]))[:k]])
    union = len(top_a | top_b)
    return len(top_a & top_b) / union if union else float("nan")


def main():
    if N_SEED_RUNS > N_SAVED_RUNS:
        raise ValueError(f"N_SEED_RUNS={N_SEED_RUNS} > N_SAVED_RUNS={N_SAVED_RUNS}")

    cache = torch.load(CACHE_PATH)
    X = cache["X"]
    y_true = np.array(cache["y_true"])
    groups = np.array(cache["groups"])

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    fold_splits = list(gkf.split(X, y_true, groups))

    # Reproduce saliency_seed_stability.py's EXACT rng call (same seed, same
    # size=10 draw) so this script's 5 seeds are a nested subset of that
    # script's 10, not an independent fresh sample.
    rng = np.random.RandomState(SAMPLE_SEED)
    all_ten = rng.choice(N_SAVED_RUNS, size=10, replace=False)
    sampled_runs = sorted(all_ten[:N_SEED_RUNS].tolist())
    print(f"Sampled {N_SEED_RUNS} seed runs (nested subset of saliency_seed_stability.py's 10): {sampled_runs}")

    pair_rows = []
    subject_rows = []
    population_region_sums = {run_idx: [] for run_idx in sampled_runs}
    all_reldeltas = []

    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        site_name = groups[test_idx[0]]
        sub_rng = np.random.RandomState(SUBJECT_SAMPLE_SEED + fold)
        sampled_local = sub_rng.choice(len(test_idx), size=min(N_SUBJECTS_PER_FOLD, len(test_idx)), replace=False)
        sampled_idx = test_idx[sampled_local]
        print(f"\nFold {fold} ({site_name}): sampling {len(sampled_idx)}/{len(test_idx)} subjects, "
              f"{N_SEED_RUNS} seeds each (IG, N_STEPS={N_STEPS})")

        backbone = build_frozen_backbone()
        test_loader = DataLoader(
            ASDVolumeDataset(X[test_idx], y_true[test_idx]), batch_size=ADABN_BATCH_SIZE
        )
        adapt_batchnorm_to_domain(backbone, test_loader)

        heads = {run_idx: load_head_for_run(run_idx, fold) for run_idx in sampled_runs}

        for local_i, idx in enumerate(sampled_idx):
            path = X[idx]
            true_label = int(y_true[idx])
            sub_id = os.path.basename(os.path.dirname(path))

            vol_np = load_and_preprocess(path, random_sigma=False)

            maps, probs, reldeltas = {}, {}, {}
            for run_idx in sampled_runs:
                model = MedNetWithHead(backbone, heads[run_idx]).to(DEVICE)
                model.eval()
                attr, prob, delta, reldelta = compute_ig(model, vol_np)
                maps[run_idx] = realign_to_mask_space(attr)
                probs[run_idx] = prob
                reldeltas[run_idx] = reldelta
                all_reldeltas.append(reldelta)

            subj_masks = load_subject_region_masks(sub_id, maps[sampled_runs[0]].shape)
            mask = foreground_mask_from_cache(subj_masks, maps[sampled_runs[0]].shape)
            n_fg = int(mask.sum())

            region_vecs = {run_idx: region_vector_from_cache(maps[run_idx], subj_masks) for run_idx in sampled_runs}
            for run_idx, vec in region_vecs.items():
                population_region_sums[run_idx].append(vec)

            if local_i < SAVE_EXAMPLE_MAPS_PER_FOLD:
                for run_idx in sampled_runs:
                    np.save(
                        os.path.join(EXAMPLES_DIR, f"{sub_id}_run{run_idx}_ig.npy"),
                        maps[run_idx].astype(np.float32),
                    )

            subj_voxel_p, subj_region_p, subj_region_pval, subj_jacc, subj_shift_pval = [], [], [], [], []
            for i in range(len(sampled_runs)):
                for j in range(i + 1, len(sampled_runs)):
                    run_i, run_j = sampled_runs[i], sampled_runs[j]
                    a, b = maps[run_i], maps[run_j]

                    vp, vs = safe_corr(a[mask], b[mask]) if n_fg >= 2 else (float("nan"), float("nan"))
                    rp, rp_pval, rs, rs_pval, n_valid_regions = region_pair_corr(region_vecs[run_i], region_vecs[run_j])
                    jacc = topk_jaccard(region_vecs[run_i], region_vecs[run_j])
                    max_reldelta = max(reldeltas[run_i], reldeltas[run_j])

                    shift_seed = hash((sub_id, run_i, run_j)) % (2**31)
                    shift_obs, shift_p, shift_null_mean, shift_null_std = (
                        shift_permutation_pvalue(a, b, mask, seed=shift_seed) if n_fg >= 2
                        else (float("nan"), float("nan"), float("nan"), float("nan"))
                    )

                    pair_rows.append({
                        "subject": sub_id, "fold": fold, "true_label": true_label,
                        "run_i": run_i, "run_j": run_j, "n_foreground_voxels": n_fg,
                        "voxel_pearson_r": vp, "voxel_spearman_r": vs,
                        "voxel_pearson_r_downsampled": shift_obs,
                        "voxel_shift_null_pvalue": shift_p,
                        "voxel_shift_null_mean_abs_r": shift_null_mean,
                        "voxel_shift_null_std_abs_r": shift_null_std,
                        "region_pearson_r": rp, "region_pearson_pvalue": rp_pval,
                        "region_spearman_r": rs, "region_spearman_pvalue": rs_pval,
                        "n_valid_regions": n_valid_regions,
                        f"top{K_TOP_REGIONS}_region_jaccard": jacc,
                        "max_relative_convergence_delta": max_reldelta,
                    })
                    subj_voxel_p.append(vp); subj_region_p.append(rp); subj_region_pval.append(rp_pval)
                    subj_jacc.append(jacc); subj_shift_pval.append(shift_p)

            def nanmeanstd(vals):
                arr = np.array(vals, dtype=np.float64)
                if np.all(np.isnan(arr)):
                    return float("nan"), float("nan")
                return float(np.nanmean(arr)), float(np.nanstd(arr))

            def fisher_combine(pvalues):
                p = np.array([x for x in pvalues if not np.isnan(x)])
                p = np.clip(p, 1e-300, 1.0)
                if len(p) == 0:
                    return float("nan")
                _, p_combined = combine_pvalues(p, method="fisher")
                return float(p_combined)

            vp_m, vp_s = nanmeanstd(subj_voxel_p)
            rp_m, rp_s = nanmeanstd(subj_region_p)
            jm, js = nanmeanstd(subj_jacc)
            omni_p = fisher_combine(subj_region_pval)
            omni_shift_p = fisher_combine(subj_shift_pval)

            subject_rows.append({
                "subject": sub_id, "fold": fold, "site": site_name, "true_label": true_label,
                "n_seed_pairs": len(subj_voxel_p), "n_foreground_voxels": n_fg,
                "mean_voxel_pearson_r": vp_m, "std_voxel_pearson_r": vp_s,
                "fisher_combined_voxel_shift_null_pvalue": omni_shift_p,
                "mean_region_pearson_r": rp_m, "std_region_pearson_r": rp_s,
                "fisher_combined_region_pearson_pvalue": omni_p,
                f"mean_top{K_TOP_REGIONS}_jaccard": jm, f"std_top{K_TOP_REGIONS}_jaccard": js,
                "max_relative_convergence_delta": max(reldeltas.values()),
            })
            print(f"  {sub_id}: mean voxel r={vp_m:.3f} (shift-null p={omni_shift_p:.2e})  "
                  f"mean region r={rp_m:.3f} (Fisher p={omni_p:.2e})  "
                  f"mean top{K_TOP_REGIONS} jaccard={jm:.3f}  max reldelta={max(reldeltas.values()):.3f}  "
                  f"({n_fg} foreground voxels)")

        del backbone, heads
        torch.cuda.empty_cache()

    with open(PAIR_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)
    print(f"\nWrote {len(pair_rows)} seed-pair rows to {PAIR_CSV}")

    with open(SUBJECT_SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(subject_rows[0].keys()))
        writer.writeheader()
        writer.writerows(subject_rows)
    print(f"Wrote {len(subject_rows)} per-subject summary rows to {SUBJECT_SUMMARY_CSV}")

    pop_vecs = {
        run_idx: np.nanmean(np.stack(vecs, axis=0), axis=0)
        for run_idx, vecs in population_region_sums.items()
    }
    pop_rows = []
    for i in range(len(sampled_runs)):
        for j in range(i + 1, len(sampled_runs)):
            run_i, run_j = sampled_runs[i], sampled_runs[j]
            rp, rp_pval, rs, rs_pval, n_valid_regions = region_pair_corr(pop_vecs[run_i], pop_vecs[run_j])
            jacc = topk_jaccard(pop_vecs[run_i], pop_vecs[run_j])
            pop_rows.append({
                "run_i": run_i, "run_j": run_j,
                "population_region_pearson_r": rp, "population_region_pearson_pvalue": rp_pval,
                "population_region_spearman_r": rs, "population_region_spearman_pvalue": rs_pval,
                "n_valid_regions": n_valid_regions,
                f"population_top{K_TOP_REGIONS}_jaccard": jacc,
            })
    with open(POPULATION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pop_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pop_rows)
    print(f"Wrote {len(pop_rows)} population-level seed-pair rows to {POPULATION_CSV}")

    all_vp = np.array([r["mean_voxel_pearson_r"] for r in subject_rows])
    all_rp = np.array([r["mean_region_pearson_r"] for r in subject_rows])
    all_j = np.array([r[f"mean_top{K_TOP_REGIONS}_jaccard"] for r in subject_rows])
    all_omni_p = np.array([r["fisher_combined_region_pearson_pvalue"] for r in subject_rows])
    all_shift_p = np.array([r["fisher_combined_voxel_shift_null_pvalue"] for r in subject_rows])
    pop_rp = np.array([r["population_region_pearson_r"] for r in pop_rows])
    pop_rp_pval = np.array([r["population_region_pearson_pvalue"] for r in pop_rows])
    reldeltas_arr = np.array(all_reldeltas)
    n_high_delta = int((reldeltas_arr > 0.1).sum())

    def fisher_combine(pvalues):
        p = np.array([x for x in pvalues if not np.isnan(x)])
        p = np.clip(p, 1e-300, 1.0)
        if len(p) == 0:
            return float("nan")
        _, p_combined = combine_pvalues(p, method="fisher")
        return float(p_combined)

    print("\n===== IG saliency seed-stability summary =====")
    print(f"{N_SEED_RUNS} seed runs per fold (nested subset of the vanilla-gradient run's 10); "
          f"{len(subject_rows)} subjects total, {len(pair_rows)} seed pairs total")
    print(f"Per-subject mean voxel Pearson r:  mean={np.nanmean(all_vp):.4f}  std={np.nanstd(all_vp):.4f}")
    print(f"Fisher-combined voxel shift-null p-value, pooling ALL subjects x seed-pairs: {fisher_combine(all_shift_p):.3e}")
    print(f"Per-subject mean region Pearson r: mean={np.nanmean(all_rp):.4f}  std={np.nanstd(all_rp):.4f}")
    print(f"Per-subject mean top{K_TOP_REGIONS} region jaccard: mean={np.nanmean(all_j):.4f}  std={np.nanstd(all_j):.4f}")
    print(f"Population-level region Pearson r across seed pairs: mean={np.nanmean(pop_rp):.4f}  std={np.nanstd(pop_rp):.4f}  "
          f"(p-values: {np.array2string(pop_rp_pval, precision=2)})")
    print(f"Fisher-combined region-r p-value, pooling ALL subjects x seed-pairs: {fisher_combine(all_omni_p):.3e}")


if __name__ == "__main__":
    main()