import csv

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split

from data_utils import DEVICE

CACHE_PATH = "/path/to/cached_features_fullres.pt"
OUT_CSV = "/path/to/null_distribution_fullres.csv"

N_PERMUTATIONS = 1000
SEED_BASE = 20240101

OBSERVED_MEAN = 0.6092

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
    return head


@torch.no_grad()
def predict(head, feat):
    head.eval()
    probs = torch.sigmoid(head(feat.to(DEVICE)).squeeze(1)).cpu().numpy()
    return (probs > 0.5).astype(float)


results = []
for p in range(N_PERMUTATIONS):
    rng = np.random.RandomState(SEED_BASE + p)
    y_perm = rng.permutation(y_true)

    fold_bal_accs = []
    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        ytrain_full = y_perm[train_idx]
        xtrain_idx, xval_idx, ytrain, yval = train_test_split(
            train_idx, ytrain_full, test_size=0.2, stratify=ytrain_full, random_state=0
        )
        ytest = y_perm[test_idx]

        head = train_head(
            raw_features[xtrain_idx], ytrain, raw_features[xval_idx], yval,
            seed=SEED_BASE + p * 10 + fold,
        )
        preds = predict(head, adapted_features[test_idx])
        fold_bal_accs.append(balanced_accuracy_score(ytest, preds))

    perm_mean = float(np.mean(fold_bal_accs))
    results.append({
        "permutation": p, "mean_balanced_acc": perm_mean,
        **{f"fold{f+1}_balanced_acc": a for f, a in enumerate(fold_bal_accs)},
    })
    if (p + 1) % 10 == 0:
        print(f"Permutation {p + 1}/{N_PERMUTATIONS}: mean balanced acc = {perm_mean:.4f}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

null_means = np.array([r["mean_balanced_acc"] for r in results])
p_value = (np.sum(null_means >= OBSERVED_MEAN) + 1) / (N_PERMUTATIONS + 1)

print("\n===== Permutation test summary =====")
print(f"Observed mean balanced accuracy (real labels): {OBSERVED_MEAN:.4f}")
print(f"Null distribution over {N_PERMUTATIONS} permutations:")
print(f"  mean: {null_means.mean():.4f}  std: {null_means.std():.4f}")
print(f"  min:  {null_means.min():.4f}  max: {null_means.max():.4f}")
print(f"Empirical p-value: {p_value:.4f}")
print(f"Saved per-permutation results to {OUT_CSV}")