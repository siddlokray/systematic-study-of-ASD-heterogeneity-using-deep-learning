import csv
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split

from data_utils import DEVICE

CACHE_PATH = "path/to/cached_features_fullres.pt"
OUT_CSV = "/path/to/seed_stability_fullres.csv"

ALL_HEADS_DIR = "/path/to/all_runs"
BEST_RUN_DIR = "path/tobest_run"
MEDIAN_RUN_DIR = "/path/to/median_run"
LOWEST_VAL_LOSS_DIR = "/path/to/lowest_val_loss"

N_RUNS = 1000
SEED_BASE = 900001

LR = 3e-5
EPOCHS = 1000
PATIENCE = 8
MIN_EPOCHS = 10
BATCH_SIZE = 4

cache = torch.load(CACHE_PATH)
X = cache["X"]
y_true = np.array(cache["y_true"])
groups = np.array(cache["groups"])
raw_features = cache["raw_features"]
adapted_features = cache["adapted_features"]

gkf = GroupKFold(n_splits=len(np.unique(groups)))
fold_splits = list(gkf.split(X, y_true, groups))

os.makedirs(ALL_HEADS_DIR, exist_ok=True)


def make_pos_weight_np(ytrain):
    n_pos = max((ytrain == 1).sum(), 1)
    n_neg = max((ytrain == 0).sum(), 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)


def train_head(xtrain_feat, ytrain, xval_feat, yval, seed):
    torch.manual_seed(seed)
    head = nn.Linear(2048, 1).to(DEVICE)
    pos_weight = make_pos_weight_np(ytrain)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)

    xtrain_t = xtrain_feat.to(DEVICE)
    ytrain_t = torch.tensor(ytrain, dtype=torch.float32, device=DEVICE)
    xval_t = xval_feat.to(DEVICE)
    yval_t = torch.tensor(yval, dtype=torch.float32, device=DEVICE)
    n = xtrain_t.shape[0]

    best_val_loss, epochs_no_improve, best_state = float("inf"), 0, None
    for epoch in range(EPOCHS):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = head(xtrain_t[idx]).squeeze(1)
            loss = criterion(logits, ytrain_t[idx])
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_loss = criterion(head(xval_t).squeeze(1), yval_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE and epoch + 1 >= MIN_EPOCHS:
                break

    head.load_state_dict(best_state)
    return head, best_val_loss


@torch.no_grad()
def predict(head, feat):
    head.eval()
    probs = torch.sigmoid(head(feat.to(DEVICE)).squeeze(1)).cpu().numpy()
    return (probs > 0.5).astype(float)


results = []
for run in range(N_RUNS):
    split_seed = SEED_BASE + run
    head_seed = SEED_BASE + run * 10 + 1

    fold_bal_accs = []
    fold_val_losses = []
    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        ytrain_full = y_true[train_idx]
        xtrain_idx, xval_idx, ytrain, yval = train_test_split(
            train_idx, ytrain_full, test_size=0.2, stratify=ytrain_full,
            random_state=split_seed,
        )
        ytest = y_true[test_idx]

        head, val_loss = train_head(
            raw_features[xtrain_idx], ytrain, raw_features[xval_idx], yval,
            seed=head_seed + fold,
        )
        preds = predict(head, adapted_features[test_idx])
        fold_bal_accs.append(balanced_accuracy_score(ytest, preds))
        fold_val_losses.append(val_loss)

        torch.save(
            head.state_dict(),
            os.path.join(ALL_HEADS_DIR, f"run{run}_fold{fold}.pt"),
        )

    run_mean = float(np.mean(fold_bal_accs))
    results.append({
        "run": run, "split_seed": split_seed, "head_seed": head_seed,
        "mean_balanced_acc": run_mean,
        **{f"fold{f + 1}_balanced_acc": a for f, a in enumerate(fold_bal_accs)},
        **{f"fold{f + 1}_val_loss": v for f, v in enumerate(fold_val_losses)},
    })
    print(f"Run {run + 1}/{N_RUNS}: mean balanced acc = {run_mean:.4f} "
          f"(per-fold: {[round(a, 3) for a in fold_bal_accs]})")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

run_means = np.array([r["mean_balanced_acc"] for r in results])
best_run_idx = int(np.argmax(run_means))
median_run_idx = int(np.argmin(np.abs(run_means - np.median(run_means))))

os.makedirs(BEST_RUN_DIR, exist_ok=True)
os.makedirs(MEDIAN_RUN_DIR, exist_ok=True)
os.makedirs(LOWEST_VAL_LOSS_DIR, exist_ok=True)

lowest_val_loss_runs = []
lowest_val_loss_accs = []
lowest_val_loss_losses = []
for fold in range(len(fold_splits)):
    shutil.copy(
        os.path.join(ALL_HEADS_DIR, f"run{best_run_idx}_fold{fold}.pt"),
        os.path.join(BEST_RUN_DIR, f"fold{fold}.pt"),
    )
    shutil.copy(
        os.path.join(ALL_HEADS_DIR, f"run{median_run_idx}_fold{fold}.pt"),
        os.path.join(MEDIAN_RUN_DIR, f"fold{fold}.pt"),
    )

    fold_val_losses_across_runs = np.array([r[f"fold{fold + 1}_val_loss"] for r in results])
    best_val_run_idx = int(np.argmin(fold_val_losses_across_runs))
    lowest_val_loss_runs.append(best_val_run_idx)
    lowest_val_loss_accs.append(results[best_val_run_idx][f"fold{fold + 1}_balanced_acc"])
    lowest_val_loss_losses.append(results[best_val_run_idx][f"fold{fold + 1}_val_loss"])
    shutil.copy(
        os.path.join(ALL_HEADS_DIR, f"run{best_val_run_idx}_fold{fold}.pt"),
        os.path.join(LOWEST_VAL_LOSS_DIR, f"fold{fold}.pt"),
    )

fold_site_names = [groups[test_idx[0]] for _, test_idx in fold_splits]

print("\n===== Seed stability summary (REAL labels) =====")
print(f"Across {N_RUNS} runs (varying split + head init):")
print(f"  mean: {run_means.mean():.4f}  std: {run_means.std():.4f}")
print(f"  min:  {run_means.min():.4f}  max: {run_means.max():.4f}")
print(f"Saved per-run results to {OUT_CSV}")
print(f"All {N_RUNS} runs' heads saved to {ALL_HEADS_DIR}")

# not used
best_run_fold_accs = [results[best_run_idx][f"fold{f + 1}_balanced_acc"] for f in range(len(fold_splits))]
print(f"\nBest run (run {best_run_idx}) copied to {BEST_RUN_DIR}")
for site, acc in zip(fold_site_names, best_run_fold_accs):
    print(f"    {site}: balanced acc = {acc:.4f}")
print(f"    mean: {np.mean(best_run_fold_accs):.4f}")

# also not used
median_run_fold_accs = [results[median_run_idx][f"fold{f + 1}_balanced_acc"] for f in range(len(fold_splits))]
print(f"\nMedian run (run {median_run_idx}) copied to {MEDIAN_RUN_DIR}")
for site, acc in zip(fold_site_names, median_run_fold_accs):
    print(f"    {site}: balanced acc = {acc:.4f}")
print(f"    mean: {np.mean(median_run_fold_accs):.4f}")

# used for saliency and analysis
print(f"\nLowest-validation-loss heads (per fold, from runs {lowest_val_loss_runs}) "
      f"copied to {LOWEST_VAL_LOSS_DIR}")
for site, run_idx, acc, vloss in zip(fold_site_names, lowest_val_loss_runs, lowest_val_loss_accs, lowest_val_loss_losses):
    print(f"    {site}: balanced acc = {acc:.4f}  (val_loss = {vloss:.4f}, from run {run_idx})")
print(f"    mean: {np.mean(lowest_val_loss_accs):.4f}")