import csv
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split
import os
import pandas as pd

from data_utils import DEVICE

CACHE_PATH = "/path/to/cached_features_fullres.pt"
QC_TABLE_PATH = "/path/to/qc_table.csv"
OUT_CSV = "/path/to/euler_seed_stability.csv"

N_RUNS = 1000
SEED_BASE = 900001

LR = 3e-5
EPOCHS = 1000
PATIENCE = 8
MIN_EPOCHS = 10
BATCH_SIZE = 4

N_PERM_PER_FOLD = 1000
SEVERITY_OBSERVED_MEAN = 0.6092

def load_qc_table(path=QC_TABLE_PATH):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df["subject"] = df["subject"].astype(int)
    return df.set_index("subject")


def subject_id_from_volume_path(path):
    return int(os.path.basename(os.path.dirname(path)))

cache = torch.load(CACHE_PATH)
X = cache["X"]
groups = np.array(cache["groups"])
raw_features = cache["raw_features"]
adapted_features = cache["adapted_features"]

qc = load_qc_table()
subj_ids = np.array([subject_id_from_volume_path(p) for p in X])
have_qc = np.array([sid in qc.index for sid in subj_ids])
recon_ok = np.array([bool(qc.loc[sid, "recon_finished"]) if sid in qc.index else False for sid in subj_ids])
keep = have_qc & recon_ok
print(f"Dropping {(~keep).sum()}/{len(X)} subjects with no/failed QC row.")

kept_idx = np.where(keep)[0]
groups_k = groups[kept_idx]
raw_features_k = raw_features[kept_idx]
adapted_features_k = adapted_features[kept_idx]
euler_total = np.array([qc.loc[sid, "euler_total"] for sid in subj_ids[kept_idx]], dtype=np.float64)

gkf = GroupKFold(n_splits=len(np.unique(groups_k)))
fold_splits = list(gkf.split(raw_features_k, euler_total, groups_k)) 


def train_head(xtrain_feat, etrain_z, xval_feat, eval_z, seed):
    torch.manual_seed(seed)
    head = nn.Linear(2048, 1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)

    xtrain_t = xtrain_feat.to(DEVICE)
    etrain_t = torch.tensor(etrain_z, dtype=torch.float32, device=DEVICE)
    xval_t = xval_feat.to(DEVICE)
    eval_t = torch.tensor(eval_z, dtype=torch.float32, device=DEVICE)
    n = xtrain_t.shape[0]

    best_val_loss, epochs_no_improve, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            preds = head(xtrain_t[idx]).squeeze(1)
            loss = criterion(preds, etrain_t[idx])
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_loss = criterion(head(xval_t).squeeze(1), eval_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE and epoch + 1 >= MIN_EPOCHS:
                break

    head.load_state_dict(best_state)
    return head


@torch.no_grad()
def predict(head, feat):
    head.eval()
    return head(feat.to(DEVICE)).squeeze(1).cpu().numpy()


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def cosine_perm_pvalue(preds, true, n_perm, seed):
    rng = np.random.default_rng(seed)
    preds = np.asarray(preds, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    observed = cosine_similarity(preds, true)

    perm_true = np.tile(true, (n_perm, 1))
    rng.permuted(perm_true, axis=1, out=perm_true)
    num = perm_true @ preds
    denom = np.linalg.norm(perm_true, axis=1) * np.linalg.norm(preds)
    denom[denom == 0] = 1e-12
    perm_vals = num / denom

    p = (np.sum(np.abs(perm_vals) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, p


results = []
for run in range(N_RUNS):
    split_seed = SEED_BASE + run
    head_seed = SEED_BASE + run * 10 + 1

    fold_cos, fold_cos_p, fold_bal_acc = [], [], []
    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        etrain_full = euler_total[train_idx]
        xtrain_idx, xval_idx, etrain, eval_ = train_test_split(
            train_idx, etrain_full, test_size=0.2, random_state=split_seed
        )
        etest = euler_total[test_idx]

        e_mean, e_std = etrain.mean(), etrain.std() + 1e-8
        etrain_z = (etrain - e_mean) / e_std
        eval_z = (eval_ - e_mean) / e_std

        head = train_head(
            raw_features_k[xtrain_idx], etrain_z, raw_features_k[xval_idx], eval_z,
            seed=head_seed + fold,
        )
        preds_z = predict(head, adapted_features_k[test_idx])
        preds_euler = preds_z * e_std + e_mean

        cos_sim, cos_p = cosine_perm_pvalue(
            preds_euler, etest, n_perm=N_PERM_PER_FOLD, seed=head_seed + fold + 500000
        )
        fold_cos.append(cos_sim)
        fold_cos_p.append(cos_p)

        median_thresh = np.median(etrain)
        true_bin = (etest > median_thresh).astype(int)
        pred_bin = (preds_euler > median_thresh).astype(int)
        fold_bal_acc.append(balanced_accuracy_score(true_bin, pred_bin))

    run_mean_cos = float(np.mean(fold_cos))
    run_mean_bal_acc = float(np.mean(fold_bal_acc))
    _, run_combined_p = stats.combine_pvalues(fold_cos_p, method="fisher")
    run_combined_p = float(run_combined_p)

    results.append({
        "run": run, "split_seed": split_seed, "head_seed": head_seed,
        "mean_cosine_sim": run_mean_cos, "combined_cosine_p": run_combined_p,
        "mean_balanced_acc": run_mean_bal_acc,
        **{f"fold{f + 1}_cos_sim": c for f, c in enumerate(fold_cos)},
        **{f"fold{f + 1}_cos_p": p for f, p in enumerate(fold_cos_p)},
        **{f"fold{f + 1}_bal_acc": a for f, a in enumerate(fold_bal_acc)},
    })
    print(
        f"Run {run + 1}/{N_RUNS}: mean cosine sim = {run_mean_cos:.4f}  "
        f"combined p = {run_combined_p:.4g}  mean balanced acc = {run_mean_bal_acc:.4f}"
    )

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

cos_vals = np.array([r["mean_cosine_sim"] for r in results])
p_vals = np.array([r["combined_cosine_p"] for r in results])
acc_vals = np.array([r["mean_balanced_acc"] for r in results])
frac_sig = float(np.mean(p_vals < 0.05))

print("\n===== Euler-prediction seed stability (REAL euler_total, frozen backbone) =====")
print(f"Across {N_RUNS} runs (varying split + head init):")
print(f"  Cosine similarity (raw, uncentered): mean={cos_vals.mean():.4f}  std={cos_vals.std():.4f}  "
      f"min={cos_vals.min():.4f}  max={cos_vals.max():.4f}")
print(f"  Fisher-combined p-value per run (test-fold permutation null): "
      f"mean={p_vals.mean():.4g}  median={np.median(p_vals):.4g}")
print(f"  Fraction of runs with combined p < 0.05: {frac_sig:.3f}")
print(f"  Balanced accuracy: mean={acc_vals.mean():.4f}  std={acc_vals.std():.4f}  "
      f"min={acc_vals.min():.4f}  max={acc_vals.max():.4f}")

t_stat, t_p = stats.ttest_1samp(acc_vals, SEVERITY_OBSERVED_MEAN)
cohens_d = (acc_vals.mean() - SEVERITY_OBSERVED_MEAN) / acc_vals.std(ddof=1)

print("\n===== t-test: euler balanced accuracy vs. severity's observed 0.6092 =====")
print(f"H0: the mean of the {N_RUNS} euler-run balanced accuracies equals {SEVERITY_OBSERVED_MEAN:.4f}")
print(f"Euler balanced accuracy: mean={acc_vals.mean():.4f}  std={acc_vals.std(ddof=1):.4f}  n={N_RUNS}")
print(f"t({N_RUNS - 1}) = {t_stat:.3f}, two-sided p = {t_p:.4g}, Cohen's d = {cohens_d:.3f}")
print(f"Saved per-run results to {OUT_CSV}")