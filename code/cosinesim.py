import os
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from average_saliency_by_region import REGION_CODES, FREESURFER_DIR

RAND_EXAMPLES_DIR = "/path/to/example_maps"
RAND_OUT_CSV = "/path/to/cosine_similarity.csv"

SEED_EXAMPLES_DIR = "/path/to/example_maps"
SEED_OUT_CSV = "/path/to/cosine_similarity.csv"


def load_region_masks(subject, shape):
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


def union_mask(masks, shape):
    fg = np.zeros(shape, dtype=bool)
    for m in masks.values():
        fg |= m
    return fg


def region_vector(sal, masks):
    vec = np.full(len(REGION_CODES), np.nan, dtype=np.float64)
    for i, region_code in enumerate(REGION_CODES):
        m = masks.get(region_code)
        if m is not None and m.any():
            vec[i] = float(sal[m].mean())
    return vec


def cosine_similarity(a, b):
    if a.size < 2:
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def abs_cosine_similarity(a, b):
    return cosine_similarity(np.abs(a), np.abs(b))


def pearson_via_centered_cosine(a, b):
    return cosine_similarity(a - a.mean(), b - b.mean())


def region_cosine_and_pearson(vec_a, vec_b):
    valid = ~(np.isnan(vec_a) | np.isnan(vec_b))
    n_valid = int(valid.sum())
    if n_valid < 2:
        return float("nan"), float("nan"), n_valid
    a, b = vec_a[valid], vec_b[valid]
    return cosine_similarity(a, b), pearson_via_centered_cosine(a, b), n_valid


DOWNSAMPLE_FACTOR = 0.5
N_SHIFTS = 200
N_REGION_PERMUTATIONS = 1000


def voxel_shift_null_cosine(map_a, map_b, mask, seed, n_shifts=N_SHIFTS, downsample=DOWNSAMPLE_FACTOR):
    a_small = zoom(map_a, downsample, order=1)
    b_small = zoom(map_b, downsample, order=1)
    mask_small = zoom(mask.astype(np.float32), downsample, order=0) > 0.5
    a_vals = a_small[mask_small]
    b_vals = b_small[mask_small]
    if a_vals.size < 2 or np.std(a_vals) == 0 or np.std(b_vals) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    obs_cos = cosine_similarity(a_vals, b_vals)
    obs_abs_cos = abs_cosine_similarity(a_vals, b_vals)

    rng = np.random.RandomState(seed)
    null_cos, null_abs_cos = [], []
    for _ in range(n_shifts):
        shift = tuple(rng.randint(1, s) for s in b_small.shape)
        shifted = np.roll(b_small, shift=shift, axis=(0, 1, 2))
        b_v = shifted[mask_small]
        if np.std(b_v) == 0:
            continue
        null_cos.append(cosine_similarity(a_vals, b_v))
        null_abs_cos.append(abs_cosine_similarity(a_vals, b_v))
    null_cos, null_abs_cos = np.array(null_cos), np.array(null_abs_cos)
    if len(null_cos) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    p_cos = float((np.sum(np.abs(null_cos) >= abs(obs_cos)) + 1) / (len(null_cos) + 1))
    p_abs_cos = float((np.sum(null_abs_cos >= obs_abs_cos) + 1) / (len(null_abs_cos) + 1))
    return p_cos, p_abs_cos, float(null_cos.mean()), float(null_abs_cos.mean())


def region_permutation_pvalue_cosine(vec_a, vec_b, n_permutations=N_REGION_PERMUTATIONS, seed=0):
    valid = ~(np.isnan(vec_a) | np.isnan(vec_b))
    n_valid = int(valid.sum())
    if n_valid < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), n_valid
    a, b = vec_a[valid], vec_b[valid]

    obs_cos = cosine_similarity(a, b)
    obs_abs_cos = abs_cosine_similarity(a, b)

    rng = np.random.RandomState(seed)
    null_cos, null_abs_cos = [], []
    for _ in range(n_permutations):
        b_perm = rng.permutation(b)
        null_cos.append(cosine_similarity(a, b_perm))
        null_abs_cos.append(abs_cosine_similarity(a, b_perm))
    null_cos, null_abs_cos = np.array(null_cos), np.array(null_abs_cos)

    p_cos = float((np.sum(np.abs(null_cos) >= abs(obs_cos)) + 1) / (len(null_cos) + 1))
    p_abs_cos = float((np.sum(null_abs_cos >= obs_abs_cos) + 1) / (len(null_abs_cos) + 1))
    return p_cos, p_abs_cos, float(null_cos.mean()), float(null_abs_cos.mean()), n_valid


RAND_STAGES = ["head", "layer4", "full"]  


def main_randomization():
    baseline_files = sorted(f for f in os.listdir(RAND_EXAMPLES_DIR) if f.endswith("_baseline_ig.npy"))
    subjects = [f[: -len("_baseline_ig.npy")] for f in baseline_files]
    print(f"Found {len(subjects)} subjects with saved baseline maps")

    rows = []
    for sub in subjects:
        base_path = os.path.join(RAND_EXAMPLES_DIR, f"{sub}_baseline_ig.npy")
        base_map = np.load(base_path)

        masks = load_region_masks(sub, base_map.shape)
        mask = union_mask(masks, base_map.shape)
        n_fg = int(mask.sum())
        if n_fg < 2:
            print(f"  {sub}: <2 foreground voxels, skipping")
            continue
        region_vec_base = region_vector(base_map, masks)

        for stage in RAND_STAGES:
            stage_path = os.path.join(RAND_EXAMPLES_DIR, f"{sub}_{stage}_randomized_ig.npy")
            stage_map = np.load(stage_path)

            a, b = base_map[mask], stage_map[mask]
            cos = cosine_similarity(a, b)
            pear_via_cos = pearson_via_centered_cosine(a, b)
            abs_cos = abs_cosine_similarity(a, b)
            p_cos, p_abs_cos, null_mean_cos, null_mean_abs_cos = voxel_shift_null_cosine(
                base_map, stage_map, mask, seed=hash((sub, stage)) % (2**31)
            )

            region_vec_stage = region_vector(stage_map, masks)
            region_cos, region_pear, n_valid_regions = region_cosine_and_pearson(region_vec_base, region_vec_stage)
            region_abs_cos = abs_cosine_similarity(
                region_vec_base[~(np.isnan(region_vec_base) | np.isnan(region_vec_stage))],
                region_vec_stage[~(np.isnan(region_vec_base) | np.isnan(region_vec_stage))],
            ) if n_valid_regions >= 2 else float("nan")
            region_p_cos, region_p_abs_cos, region_null_mean_cos, region_null_mean_abs_cos, _ = (
                region_permutation_pvalue_cosine(region_vec_base, region_vec_stage,
                                                  seed=hash((sub, stage, "region")) % (2**31))
            )

            rows.append({
                "subject": sub, "comparison": f"baseline_vs_{stage}", "n_foreground_voxels": n_fg,
                "voxel_cosine_similarity": cos, "voxel_pearson_r_via_centered_cosine": pear_via_cos,
                "voxel_abs_cosine_similarity": abs_cos,
                "voxel_cosine_shift_null_pvalue": p_cos, "voxel_abs_cosine_shift_null_pvalue": p_abs_cos,
                "voxel_cosine_null_mean": null_mean_cos, "voxel_abs_cosine_null_mean": null_mean_abs_cos,
                "region_cosine_similarity": region_cos, "region_pearson_r_via_centered_cosine": region_pear,
                "region_abs_cosine_similarity": region_abs_cos,
                "region_cosine_permutation_pvalue": region_p_cos,
                "region_abs_cosine_permutation_pvalue": region_p_abs_cos,
                "n_valid_regions": n_valid_regions,
            })
            print(f"  {sub}/{stage}: voxel cosine={cos:.4f} (p={p_cos:.3f}) abs_cosine={abs_cos:.4f} (p={p_abs_cos:.3f})   "
                  f"region cosine={region_cos:.4f} (p={region_p_cos:.3f}) abs_cosine={region_abs_cos:.4f} (p={region_p_abs_cos:.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(RAND_OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {RAND_OUT_CSV}")
    for stage in RAND_STAGES:
        g = df[df["comparison"] == f"baseline_vs_{stage}"]
        if len(g) == 0:
            continue
        print(f"\n{stage:8s}  voxel:  mean cosine={g['voxel_cosine_similarity'].mean():.4f} "
              f"(mean p={g['voxel_cosine_shift_null_pvalue'].mean():.3f})   "
              f"mean abs_cosine={g['voxel_abs_cosine_similarity'].mean():.4f} "
              f"(mean p={g['voxel_abs_cosine_shift_null_pvalue'].mean():.3f})")
        print(f"{'':8s}  region: mean cosine={g['region_cosine_similarity'].mean():.4f} "
              f"(mean p={g['region_cosine_permutation_pvalue'].mean():.3f})   "
              f"mean abs_cosine={g['region_abs_cosine_similarity'].mean():.4f} "
              f"(mean p={g['region_abs_cosine_permutation_pvalue'].mean():.3f})")
    return df


def main_seed_stability():
    all_files = sorted(f for f in os.listdir(SEED_EXAMPLES_DIR) if f.endswith("_ig.npy"))
    parsed = []
    for f in all_files:
        stem = f[: -len("_ig.npy")]
        subject, run_part = stem.rsplit("_run", 1)
        parsed.append((subject, int(run_part), f))
    subjects = sorted(set(p[0] for p in parsed))
    print(f"Found {len(subjects)} subjects with saved seed-run maps")

    maps = {}
    region_masks = {}
    fg_masks = {}
    for subject, run_idx, fname in parsed:
        maps[(subject, run_idx)] = np.load(os.path.join(SEED_EXAMPLES_DIR, fname))
        if subject not in region_masks:
            region_masks[subject] = load_region_masks(subject, maps[(subject, run_idx)].shape)
            fg_masks[subject] = union_mask(region_masks[subject], maps[(subject, run_idx)].shape)

    rows = []
    for subject in subjects:
        runs = sorted(r for (s, r) in maps if s == subject)
        mask = fg_masks[subject]
        masks = region_masks[subject]
        n_fg = int(mask.sum())
        if n_fg < 2:
            continue
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                run_i, run_j = runs[i], runs[j]
                a, b = maps[(subject, run_i)][mask], maps[(subject, run_j)][mask]
                cos = cosine_similarity(a, b)
                pear_via_cos = pearson_via_centered_cosine(a, b)
                abs_cos = abs_cosine_similarity(a, b)
                p_cos, p_abs_cos, null_mean_cos, null_mean_abs_cos = voxel_shift_null_cosine(
                    maps[(subject, run_i)], maps[(subject, run_j)], mask,
                    seed=hash((subject, run_i, run_j)) % (2**31)
                )

                region_vec_i = region_vector(maps[(subject, run_i)], masks)
                region_vec_j = region_vector(maps[(subject, run_j)], masks)
                region_cos, region_pear, n_valid_regions = region_cosine_and_pearson(region_vec_i, region_vec_j)
                valid_ij = ~(np.isnan(region_vec_i) | np.isnan(region_vec_j))
                region_abs_cos = abs_cosine_similarity(region_vec_i[valid_ij], region_vec_j[valid_ij]) \
                    if n_valid_regions >= 2 else float("nan")
                region_p_cos, region_p_abs_cos, region_null_mean_cos, region_null_mean_abs_cos, _ = (
                    region_permutation_pvalue_cosine(region_vec_i, region_vec_j,
                                                      seed=hash((subject, run_i, run_j, "region")) % (2**31))
                )

                rows.append({
                    "subject": subject, "run_i": run_i, "run_j": run_j, "n_foreground_voxels": n_fg,
                    "voxel_cosine_similarity": cos, "voxel_pearson_r_via_centered_cosine": pear_via_cos,
                    "voxel_abs_cosine_similarity": abs_cos,
                    "voxel_cosine_shift_null_pvalue": p_cos, "voxel_abs_cosine_shift_null_pvalue": p_abs_cos,
                    "region_cosine_similarity": region_cos, "region_pearson_r_via_centered_cosine": region_pear,
                    "region_abs_cosine_similarity": region_abs_cos,
                    "region_cosine_permutation_pvalue": region_p_cos,
                    "region_abs_cosine_permutation_pvalue": region_p_abs_cos,
                    "n_valid_regions": n_valid_regions,
                })

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} seed-pair rows across {len(subjects)} subjects")
    for subject, g in df.groupby("subject"):
        print(f"  {subject}: voxel cosine={g['voxel_cosine_similarity'].mean():.4f} "
              f"(p={g['voxel_cosine_shift_null_pvalue'].mean():.3f}) "
              f"abs_cosine={g['voxel_abs_cosine_similarity'].mean():.4f} "
              f"(p={g['voxel_abs_cosine_shift_null_pvalue'].mean():.3f})   "
              f"region cosine={g['region_cosine_similarity'].mean():.4f} "
              f"(p={g['region_cosine_permutation_pvalue'].mean():.3f}) "
              f"abs_cosine={g['region_abs_cosine_similarity'].mean():.4f} "
              f"(p={g['region_abs_cosine_permutation_pvalue'].mean():.3f})")

    df.to_csv(SEED_OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {SEED_OUT_CSV}")
    print(f"\nVoxel overall:  mean cosine={df['voxel_cosine_similarity'].mean():.4f}  "
          f"mean pearson={df['voxel_pearson_r_via_centered_cosine'].mean():.4f}  "
          f"mean gap={((df['voxel_cosine_similarity'] - df['voxel_pearson_r_via_centered_cosine']).mean()):+.4f}")
    print(f"Region overall: mean cosine={df['region_cosine_similarity'].mean():.4f}  "
          f"mean pearson={df['region_pearson_r_via_centered_cosine'].mean():.4f}  "
          f"mean gap={((df['region_cosine_similarity'] - df['region_pearson_r_via_centered_cosine']).mean()):+.4f}")
    return df


if __name__ == "__main__":
    print("=" * 80)
    print("RANDOMIZATION (baseline vs. fully-randomized)")
    print("=" * 80)
    df_rand = main_randomization()

    print("\n" + "=" * 80)
    print("SEED STABILITY (trained-vs-trained, different seeds)")
    print("=" * 80)
    df_seed = main_seed_stability()

    print(f"\nBoth saved: {RAND_OUT_CSV}\n            {SEED_OUT_CSV}")