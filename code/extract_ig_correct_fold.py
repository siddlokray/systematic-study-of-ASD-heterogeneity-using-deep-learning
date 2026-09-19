import csv
import os
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from data_utils import DEVICE, ASDVolumeDataset, load_and_preprocess, adapt_batchnorm_to_domain
from medicalnet import build_backbone, load_pretrained_backbone

CACHE_PATH = "/path/to/cached_features_fullres.pt"
CKPT_PATH = "/path/to/resnet_50_23dataset.pth"
MODEL_DEPTH = 50
ADABN_BATCH_SIZE = 1

LOWEST_VAL_LOSS_DIR = "/path/to/lowest_val_loss"
OUT_DIR = "path/to/ig_correct_fold"
os.makedirs(OUT_DIR, exist_ok=True)

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
        return self.head(feat)  # (batch, 1) logit


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


def make_baseline(vol):
    if BASELINE == "zero":
        return torch.zeros_like(vol)
    else:
        raise ValueError(f"Unknown BASELINE: {BASELINE}")


def compute_ig(model, path):
    vol_np = load_and_preprocess(path, random_sigma=False)
    vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    baseline = make_baseline(vol)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        vol, baselines=baseline, target=0,
        n_steps=N_STEPS, internal_batch_size=INTERNAL_BATCH_SIZE,
        return_convergence_delta=True,
    )

    with torch.no_grad():
        logit = model(vol).squeeze()

    attr = attributions.detach().cpu().numpy()[0, 0]
    return attr, torch.sigmoid(logit).item(), float(delta.abs().item())


def main():
    cache = torch.load(CACHE_PATH)
    X = cache["X"]
    y_true = np.array(cache["y_true"])
    groups = np.array(cache["groups"])

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    fold_splits = list(gkf.split(X, y_true, groups))

    manifest = []
    for fold, (train_idx, test_idx) in enumerate(fold_splits):
        site_name = groups[test_idx[0]]
        print(f"Fold {fold} ({site_name}): {len(test_idx)} subjects")

        backbone = build_frozen_backbone()
        test_loader = DataLoader(
            ASDVolumeDataset(X[test_idx], y_true[test_idx]), batch_size=ADABN_BATCH_SIZE
        )
        adapt_batchnorm_to_domain(backbone, test_loader)
        head = load_head(fold)
        model = MedNetWithHead(backbone, head).to(DEVICE)
        model.eval()

        correct = 0
        for idx in test_idx:
            path = X[idx]
            true_label = int(y_true[idx])
            attr, prob, conv_delta = compute_ig(model, path)
            attr_aligned = realign_to_mask_space(attr)

            sub_id = os.path.basename(os.path.dirname(path))
            out_path = os.path.join(OUT_DIR, f"{sub_id}_saliency.npy")
            np.save(out_path, attr_aligned.astype(np.float32))

            pred = int(prob > 0.5)
            mismatch = pred != true_label
            correct += int(not mismatch)
            manifest.append({
                "subject": sub_id, "fold": fold, "site": site_name,
                "true_label": true_label, "pred_label": pred,
                "pred_prob_high": prob, "misclassified": mismatch,
                "convergence_delta": conv_delta,
                "saliency_path": out_path,
            })
            flag = "  <-- misclassified by the model" if mismatch else ""
            print(f"  {sub_id}: true={true_label}  pred={pred}  P(high)={prob:.3f}  "
                  f"conv_delta={conv_delta:.2e}{flag}")

        print(f"  fold {fold} raw accuracy on this reconstruction: {correct}/{len(test_idx)}")

        del backbone, head, model
        torch.cuda.empty_cache()

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\nWrote {len(manifest)} IG attribution maps + manifest to {manifest_path}")


if __name__ == "__main__":
    main()
