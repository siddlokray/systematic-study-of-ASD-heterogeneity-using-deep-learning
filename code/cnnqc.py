import csv
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split
import pandas as pd

from data_utils import DEVICE

CACHE_PATH = "/path/to/cached_features_fullres.pt"
QC_TABLE_PATH = "/path/to/qc_table.csv"
OUT_SUBJECT_CSV = "/path/to/cnn_vs_qc_subject_level.csv"

LR = 3e-5
EPOCHS = 1000
PATIENCE = 8
MIN_EPOCHS = 10
BATCH_SIZE = 4

def load_qc_table(path=QC_TABLE_PATH):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    df["subject"] = df["subject"].astype(int)
    return df.set_index("subject")


def subject_id_from_volume_path(path):
    return int(os.path.basename(os.path.dirname(path)))

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
def predict_probs(head, feat):
    head.eval()
    return torch.sigmoid(head(feat.to(DEVICE)).squeeze(1)).cpu().numpy()


subj_ids = np.array([subject_id_from_volume_path(p) for p in X])
cnn_prob = np.full(len(X), np.nan)

for fold, (train_idx, test_idx) in enumerate(fold_splits):
    ytrain_full = y_true[train_idx]
    xtrain_idx, xval_idx, ytrain, yval = train_test_split(
        train_idx, ytrain_full, test_size=0.2, stratify=ytrain_full, random_state=0
    )
    head = train_head(raw_features[xtrain_idx], ytrain, raw_features[xval_idx], yval, seed=20240101 + fold)
    probs = predict_probs(head, adapted_features[test_idx])
    cnn_prob[test_idx] = probs
    print(f"Fold {fold + 1} ({groups[test_idx[0]]}): done")

assert not np.isnan(cnn_prob).any(), "Some subjects never got a fold assignment -- check fold_splits coverage."

qc = load_qc_table()
have_qc = np.array([sid in qc.index for sid in subj_ids])
recon_ok = np.array([bool(qc.loc[sid, "recon_finished"]) if sid in qc.index else False for sid in subj_ids])
keep = have_qc & recon_ok
print(f"Dropping {(~keep).sum()} subjects with no/failed QC row.")

euler_total = np.full(len(X), np.nan)
etiv = np.full(len(X), np.nan)
for i, sid in enumerate(subj_ids):
    if keep[i]:
        euler_total[i] = qc.loc[sid, "euler_total"]
        etiv[i] = qc.loc[sid, "etiv"]

rows = [
    {"subject": subj_ids[i], "site": groups[i], "true_label": int(y_true[i]),
     "cnn_prob": float(cnn_prob[i]), "euler_total": euler_total[i], "etiv": etiv[i]}
    for i in range(len(X)) if keep[i]
]
with open(OUT_SUBJECT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subject", "site", "true_label", "cnn_prob", "euler_total", "etiv"])
    w.writeheader()
    w.writerows(rows)
print(f"Saved subject-level table ({len(rows)} rows) to {OUT_SUBJECT_CSV}")

kept_idx = np.where(keep)[0]
cp = cnn_prob[kept_idx]
et = euler_total[kept_idx]
yl = y_true[kept_idx]
gl = groups[kept_idx]

pearson_r, pearson_p = stats.pearsonr(cp, et)
spearman_r, spearman_p = stats.spearmanr(cp, et)
print("\n===== CNN output vs Euler number =====")
print(f"Overall (n={len(cp)}): Pearson r={pearson_r:.3f} p={pearson_p:.4f}  "
      f"Spearman r={spearman_r:.3f} p={spearman_p:.4f}")

for label, name in [(0, "low severity"), (1, "high severity")]:
    m = yl == label
    if m.sum() > 3:
        r, p = stats.pearsonr(cp[m], et[m])
        print(f"  within {name} (n={m.sum()}): Pearson r={r:.3f} p={p:.4f}")

def run_logreg_baseline(feat, y_labels, groups_arr, seed=0):
    gkf2 = GroupKFold(n_splits=len(np.unique(groups_arr)))
    accs = []
    for tr, te in gkf2.split(feat, y_labels, groups_arr):
        clf = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)
        clf.fit(feat[tr], y_labels[tr])
        preds = clf.predict(feat[te])
        accs.append(balanced_accuracy_score(y_labels[te], preds))
    return accs

print("\n===== Joint model comparison (leave-site-out, real labels) =====")
for name, feat in [
    ("cnn_prob only", cp.reshape(-1, 1)),
    ("euler_total only", et.reshape(-1, 1)),
    ("cnn_prob + euler_total", np.column_stack([cp, et])),
]:
    accs = run_logreg_baseline(feat, yl, gl)
    print(f"{name:>24}: mean balanced acc = {np.mean(accs):.4f}  (per-fold: {[round(a, 3) for a in accs]})")