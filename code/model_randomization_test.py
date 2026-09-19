import copy
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

LOWEST_VAL_LOSS_DIR = "/path/to/lowest_val_loss"

OUT_DIR = "/path/to/model_randomization_test_ig"
EXAMPLES_DIR = os.path.join(OUT_DIR, "example_maps")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, "randomization_correlations_ig.csv")

N_SUBJECTS_PER_FOLD = 4
SUBJECT_SAMPLE_SEED = 20260817
RANDOMIZATION_SEED_BASE = 777

K_TOP_REGIONS = 5
SAVE_EXAMPLE_MAPS = True

STAGES = ["head", "layer4", "layer3", "layer2", "layer1", "stem"]
EVAL_POINTS = ["head", "layer4", "full"]

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


def load_head(fold):
    head = nn.Linear(2048, 1).to(DEVICE)
    state = torch.load(os.path.join(LOWEST_VAL_LOSS_DIR, f"fold{fold}.pt"), map_location=DEVICE)
    head.load_state_dict(state)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head


def realign_to_mask_space(sal):
    sal = np.flip(sal, axis=2)
    sal = np.transpose(sal, (0, 2, 1))
    return sal.T


def get_stage_module(model, stage_name):
    if stage_name == "head":
        return model.head
    if stage_name == "stem":
        return nn.ModuleList([model.backbone.conv1, model.backbone.bn1])
    return getattr(model.backbone, stage_name)


def stage_seed(layer_name, fold, trial=0):
    return RANDOMIZATION_SEED_BASE + fold * 100000 + STAGES.index(layer_name) * 1000 + trial * 10


def randomize_module_(module, seed):
    torch.manual_seed(seed)
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            m.reset_parameters()
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_parameters()
            m.reset_running_stats()


def build_eval_models(base_model, fold):
    models = {}

    m_head = copy.deepcopy(base_model)
    randomize_module_(get_stage_module(m_head, "head"), seed=stage_seed("head", fold))
    m_head.eval()
    models["head"] = m_head

    m_layer4 = copy.deepcopy(base_model)
    randomize_module_(get_stage_module(m_layer4, "head"), seed=stage_seed("head", fold))
    randomize_module_(get_stage_module(m_layer4, "layer4"), seed=stage_seed("layer4", fold))
    m_layer4.eval()
    models["layer4"] = m_layer4

    m_full = copy.deepcopy(base_model)
    for s in STAGES:
        randomize_module_(get_stage_module(m_full, s), seed=stage_seed(s, fold))
    m_full.eval()
    models["full"] = m_full

    return models

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
    if n_valid < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_valid
    a, b = vec_a[valid], vec_b[valid]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_valid
    pear_r, pear_p = pearsonr(a, b)
    spear_r, spear_p = spearmanr(a, b)
    return float(pear_r), float(pear_p), float(spear_r), float(spear_p), n_valid


DOWNSAMPLE_FACTOR = 0.5
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
    cache = torch.load(CACHE_PATH)
    X = cache["X"]
    y_true = np.array(cache["y_true"])
    groups = np.array(cache["groups"])

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    fold_splits = list(gkf.split(X, y_true, groups))

    rows = []

    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        site_name = groups[test_idx[0]]
        rng = np.random.RandomState(SUBJECT_SAMPLE_SEED + fold)
        sampled_local = rng.choice(len(test_idx), size=min(N_SUBJECTS_PER_FOLD, len(test_idx)), replace=False)
        sampled_idx = test_idx[sampled_local]
        print(f"\nFold {fold} ({site_name}): sampling {len(sampled_idx)}/{len(test_idx)} subjects (IG, N_STEPS={N_STEPS})")

        backbone = build_frozen_backbone()
        test_loader = DataLoader(
            ASDVolumeDataset(X[test_idx], y_true[test_idx]), batch_size=ADABN_BATCH_SIZE
        )
        adapt_batchnorm_to_domain(backbone, test_loader)
        head = load_head(fold)
        base_model = MedNetWithHead(backbone, head).to(DEVICE)
        base_model.eval()

        print("  Building head / layer4 / full randomized models...")
        eval_models = build_eval_models(base_model, fold)

        for idx in sampled_idx:
            path = X[idx]
            true_label = int(y_true[idx])
            sub_id = os.path.basename(os.path.dirname(path))
            vol_np = load_and_preprocess(path, random_sigma=False)

            base_attr, base_prob, base_delta, base_reldelta = compute_ig(base_model, vol_np)
            base_map = realign_to_mask_space(base_attr)
            pred_label = int(base_prob > 0.5)
            subj_masks = load_subject_region_masks(sub_id, base_map.shape)
            mask = foreground_mask_from_cache(subj_masks, base_map.shape)
            n_fg = int(mask.sum())
            base_region_vec = region_vector_from_cache(base_map, subj_masks)

            if SAVE_EXAMPLE_MAPS:
                np.save(os.path.join(EXAMPLES_DIR, f"{sub_id}_baseline_ig.npy"), base_map.astype(np.float32))

            print(f"  {sub_id}: true={true_label} pred={pred_label} P(high)={base_prob:.3f} "
                  f"baseline conv_delta={base_delta:.2e} (rel={base_reldelta:.3f})  ({n_fg} fg voxels)")

            per_subject_str = []
            for stage in EVAL_POINTS:
                model = eval_models[stage]
                attr, prob, delta, reldelta = compute_ig(model, vol_np)
                rmap = realign_to_mask_space(attr)

                if n_fg >= 2:
                    vp, vs = safe_corr(base_map[mask], rmap[mask])
                else:
                    vp, vs = float("nan"), float("nan")
                rvec = region_vector_from_cache(rmap, subj_masks)
                rp, rp_pval, rs, rs_pval, n_valid_regions = region_pair_corr(base_region_vec, rvec)
                jacc = topk_jaccard(base_region_vec, rvec)

                shift_seed = hash((sub_id, stage)) % (2**31)
                shift_obs, shift_p, shift_null_mean, shift_null_std = (
                    shift_permutation_pvalue(base_map, rmap, mask, seed=shift_seed) if n_fg >= 2
                    else (float("nan"), float("nan"), float("nan"), float("nan"))
                )

                rows.append({
                    "subject": sub_id, "fold": fold, "site": site_name,
                    "true_label": true_label, "pred_label": pred_label,
                    "stage": stage, "randomized_prob_high": prob,
                    "n_foreground_voxels": n_fg,
                    "voxel_pearson_r": vp, "voxel_spearman_r": vs,
                    "voxel_pearson_r_downsampled": shift_obs,
                    "voxel_shift_null_pvalue": shift_p,
                    "voxel_shift_null_mean_abs_r": shift_null_mean,
                    "voxel_shift_null_std_abs_r": shift_null_std,
                    "region_pearson_r": rp, "region_pearson_pvalue": rp_pval,
                    "region_spearman_r": rs, "region_spearman_pvalue": rs_pval,
                    "n_valid_regions": n_valid_regions,
                    f"top{K_TOP_REGIONS}_region_jaccard": jacc,
                    "convergence_delta": delta, "relative_convergence_delta": reldelta,
                })
                per_subject_str.append(f"{stage}={vp:.3f}(p={rp_pval:.2e},shift_p={shift_p:.2e},reldelta={reldelta:.2f})")

                if SAVE_EXAMPLE_MAPS:
                    np.save(os.path.join(EXAMPLES_DIR, f"{sub_id}_{stage}_randomized_ig.npy"),
                            rmap.astype(np.float32))

            print("    voxel r vs baseline: " + ", ".join(per_subject_str))

        del backbone, head, base_model, eval_models
        torch.cuda.empty_cache()

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS_CSV}")

    def fisher_combine(pvalues):
        p = np.array([x for x in pvalues if not np.isnan(x)])
        p = np.clip(p, 1e-300, 1.0)
        if len(p) == 0:
            return float("nan")
        _, p_combined = combine_pvalues(p, method="fisher")
        return float(p_combined)

    print("\n===== IG model-randomization summary (single draw, cascade only) =====")
    for stage in EVAL_POINTS:
        vals = np.array([r["voxel_pearson_r"] for r in rows if r["stage"] == stage])
        rvals = np.array([r["region_pearson_r"] for r in rows if r["stage"] == stage])
        rpvals = [r["region_pearson_pvalue"] for r in rows if r["stage"] == stage]
        shift_pvals = [r["voxel_shift_null_pvalue"] for r in rows if r["stage"] == stage]
        jvals = np.array([r[f"top{K_TOP_REGIONS}_region_jaccard"] for r in rows if r["stage"] == stage])
        reldeltas = np.array([r["relative_convergence_delta"] for r in rows if r["stage"] == stage])
        omni_p = fisher_combine(rpvals)
        omni_shift_p = fisher_combine(shift_pvals)
        print(f"  {stage:8s}  mean voxel r={np.nanmean(vals):+.4f}  mean |voxel r|={np.nanmean(np.abs(vals)):.4f}  "
              f"Fisher-combined voxel shift-null p={omni_shift_p:.3e}  "
              f"mean region r={np.nanmean(rvals):+.4f}  Fisher-combined region-r p={omni_p:.3e}  "
              f"top{K_TOP_REGIONS} jaccard={np.nanmean(jvals):.4f}  "
              f"| relative_convergence_delta: mean={np.nanmean(reldeltas):.3f} max={np.nanmax(reldeltas):.3f}")

    all_reldeltas = np.array([r["relative_convergence_delta"] for r in rows])
    n_high_delta = int((all_reldeltas > 0.1).sum())

if __name__ == "__main__":
    main()