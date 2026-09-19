import math
import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, zoom
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import Dataset
import random
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PreprocessConfig:
    volume = "nu-to-mni305.mgz"
    imsize = 256
    factor = imsize / 256
    fwhm = 3
    voxsize = 1
    sigma_range = (1.0, 1.5)


FLIST = [
    "/projectnb/nickar/freesurfer/csv/ABIDEII-NYU_1.csv",
    "/projectnb/nickar/freesurfer/csv/ABIDEII-SDSU_1.csv",
    "/projectnb/nickar/freesurfer/csv/ABIDEII-TCD_1.csv",
    "/projectnb/nickar/freesurfer/csv/ABIDEII-OHSU_1.csv",
    "/projectnb/nickar/freesurfer/csv/ABIDEII-KKI_1.csv",
]


def build_file_list(flist=FLIST, cfg=PreprocessConfig):
    f, l, site = [], [], []
    for college_csv in flist:
        csv = pd.read_csv(college_csv)
        csv = csv[["SUB_ID", "DX_GROUP", "AGE_AT_SCAN ", "SEX", "SRS_EDITION", "SRS_TOTAL_T"]]
        for i, r in csv.iterrows():
            p = os.path.join("/projectnb/nickar/freesurfer", str(int(r[0])), str(cfg.volume))
            if not os.path.exists(p):
                continue
            try:
                if math.isnan(r[5]):
                    continue
                if (r[2] <= 12) and (r[3] == 1):
                    if r[5] < 45:
                        f.append(p); l.append(0)
                        site.append(college_csv.split("/")[-1].split(".")[0].split("-")[-1])
                    elif r[5] > 70:
                        f.append(p); l.append(1)
                        site.append(college_csv.split("/")[-1].split(".")[0].split("-")[-1])
            except Exception as e:
                print(r[0], e)
    return np.array(f), np.array(l), np.array(site)


def smooth(v, fwhm, vs):
    sigma = fwhm / (np.sqrt(8 * np.log(2)) * vs)
    return gaussian_filter(v, sigma=sigma)


def smooth_sigma(v, sigma):
    return gaussian_filter(v, sigma=sigma)


def load_and_preprocess(path, cfg=PreprocessConfig, random_sigma=False):
    image = nib.load(path).get_fdata()
    image = image.T
    image = np.transpose(image, (0, 2, 1))
    image = np.flip(image, axis=2)
    x, y_, z = image.shape
    image = np.pad(image, ((0, 256 - x), (0, 256 - y_), (0, 256 - z)), mode="constant")
    if random_sigma:
        sigma = random.uniform(*cfg.sigma_range)
        image = smooth_sigma(image, sigma)
    else:
        image = smooth(image, cfg.fwhm, cfg.voxsize)
    image = zoom(image, (cfg.factor, cfg.factor, cfg.factor))
    image = (image - image.mean()) / (image.std() + 1e-8)
    return image.astype(np.float32)


class ASDVolumeDataset(Dataset):
    def __init__(self, filepaths, labels, cfg=PreprocessConfig, random_sigma=False):
        self.filepaths = filepaths
        self.labels = labels
        self.cfg = cfg
        self.random_sigma = random_sigma

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        vol = load_and_preprocess(self.filepaths[idx], self.cfg, random_sigma=self.random_sigma)
        vol = torch.from_numpy(vol).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return vol, label


def _set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def run_epoch(model, loader, criterion, optimizer=None, desc="epoch", freeze_bn=True, accum_steps=1):
    train = optimizer is not None
    model.train(train)
    if train and freeze_bn:
        _set_bn_eval(model)
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False, unit="vol")
    if train:
        optimizer.zero_grad()
    num_batches = 0
    for vols, labels in pbar:
        vols, labels = vols.to(DEVICE), labels.to(DEVICE)
        with torch.set_grad_enabled(train):
            logits = model(vols).squeeze(1)
            loss = criterion(logits, labels)
            if train:
                (loss / accum_steps).backward()
                num_batches += 1
                if num_batches % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
        total_loss += loss.item() * vols.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        n += vols.size(0)
        if hasattr(pbar, "set_postfix"):  
            pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.3f}")
    if train and num_batches % accum_steps != 0:
        optimizer.step() 
        optimizer.zero_grad()
    return total_loss / n, correct / n
    

def adapt_batchnorm_to_domain(model, loader):
    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bn_modules:
        m.reset_running_stats()  
        m.momentum = None 
        m.train()              
 
    with torch.no_grad():
        for vols, _ in loader:  
            model(vols.to(DEVICE))
 
    for m in bn_modules:
        m.eval() 


def predict_all(model, loader):
    model.train(False)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for vols, labels in loader:
            vols = vols.to(DEVICE)
            logits = model(vols).squeeze(1)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs > 0.5).float()
            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()


def report_probability_spread(name, probs):
    print(f"[{name}] predicted P(high severity): mean={probs.mean():.4f} "
          f"std={probs.std():.4f} min={probs.min():.4f} max={probs.max():.4f}")


def report_fold_diagnostics(name, y_true, y_pred=None):
    vals, counts = np.unique(y_true, return_counts=True)
    balance = dict(zip(vals.astype(int), counts))
    print(f"[{name}] class balance (0=low,1=high): {balance}")
    if y_pred is not None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        print(f"[{name}] predicted class counts: "
              f"{dict(zip(*np.unique(y_pred.astype(int), return_counts=True)))}")
        print(f"[{name}] confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")
        print(f"[{name}] balanced accuracy: {bal_acc:.4f}")


def make_pos_weight(ytrain):
    n_pos = max((ytrain == 1).sum(), 1)
    n_neg = max((ytrain == 0).sum(), 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)