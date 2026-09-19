import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from data_utils import DEVICE, build_file_list, ASDVolumeDataset, adapt_batchnorm_to_domain
from medicalnet import build_backbone, load_pretrained_backbone

CKPT_PATH = "/path/to/resnet_50_23dataset.pth"
MODEL_DEPTH = 50
BATCH_SIZE = 1
OUT_PATH = "/path/to/cached_features_fullres.pt"

X, y, groups = build_file_list()
print(f"Total subjects: {len(X)}")

pool = nn.AdaptiveAvgPool3d(1)


def build_frozen_backbone():
    backbone = build_backbone(MODEL_DEPTH)
    load_pretrained_backbone(backbone, CKPT_PATH, device=DEVICE, verbose=False)
    backbone = backbone.to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


@torch.no_grad()
def extract_one(backbone, path):
    from data_utils import load_and_preprocess
    vol = load_and_preprocess(path, random_sigma=False)  # deterministic, fwhm=3
    vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(DEVICE)
    feat = pool(backbone(vol)).flatten(1)
    return feat.squeeze(0).cpu()


print("Extracting raw (pretrained-BN) features for all subjects...")
backbone = build_frozen_backbone()
raw_features = torch.zeros(len(X), 2048)
for i, path in enumerate(X):
    raw_features[i] = extract_one(backbone, path)
    if (i + 1) % 20 == 0 or i == len(X) - 1:
        print(f"  {i + 1}/{len(X)}")
del backbone
torch.cuda.empty_cache()

gkf = GroupKFold(n_splits=len(np.unique(groups)))
adapted_features = torch.zeros(len(X), 2048)
fold_site_names = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    site_name = groups[test_idx[0]]
    fold_site_names.append(site_name)
    print(f"Fold {fold + 1} ({site_name}): AdaBN-adapting on {len(test_idx)} subjects...")

    backbone = build_frozen_backbone()
    test_loader = DataLoader(
        ASDVolumeDataset(X[test_idx], y[test_idx]), batch_size=BATCH_SIZE
    )
    adapt_batchnorm_to_domain(backbone, test_loader)

    for idx in test_idx:
        adapted_features[idx] = extract_one(backbone, X[idx])

    del backbone
    torch.cuda.empty_cache()

torch.save({
    "X": X, "y_true": y, "groups": groups,
    "raw_features": raw_features,
    "adapted_features": adapted_features,
    "fold_site_names": fold_site_names,
}, OUT_PATH)
print(f"Saved cached features to {OUT_PATH}")