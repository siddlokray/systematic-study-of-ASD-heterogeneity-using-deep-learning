import csv
import os
import nibabel as nib
import numpy as np

FREESURFER_DIR = "/path/to/freesurfer"
SALIENCY_DIR = "/path/to/ig_correct_fold"
MANIFEST_PATH = os.path.join(SALIENCY_DIR, "manifest.csv")
OUT_PATH = os.path.join(SALIENCY_DIR, "avg_ig_by_region_sub.csv")
WIDE_OUT_PATH = os.path.join(SALIENCY_DIR, "avg_ig_by_region_wide_sub.csv")

# cortical lh/rh
REGION_CODES = [
    11101, 11102, 11103, 11104, 11105, 11106, 11107, 11108, 11109, 11110, 11111, 11112, 11113, 11114, 11115,
    11116, 11117, 11118, 11119, 11120, 11121, 11122, 11123, 11124, 11125, 11126, 11127, 11128, 11129, 11130,
    11131, 11132, 11133, 11134, 11135, 11136, 11137, 11138, 11139, 11140, 11141, 11142, 11143, 11144, 11145,
    11146, 11147, 11148, 11149, 11150, 11151, 11152, 11153, 11154, 11155, 11156, 11157, 11158, 11159, 11160,
    11161, 11162, 11163, 11164, 11165, 11166, 11167, 11168, 11169, 11170, 11171, 11172, 11173, 11174, 11175,
    12101, 12102, 12103, 12104, 12105, 12106, 12107, 12108, 12109, 12110, 12111, 12112, 12113, 12114, 12115,
    12116, 12117, 12118, 12119, 12120, 12121, 12122, 12123, 12124, 12125, 12126, 12127, 12128, 12129, 12130,
    12131, 12132, 12133, 12134, 12135, 12136, 12137, 12138, 12139, 12140, 12141, 12142, 12143, 12144, 12145,
    12146, 12147, 12148, 12149, 12150, 12151, 12152, 12153, 12154, 12155, 12156, 12157, 12158, 12159, 12160,
    12161, 12162, 12163, 12164, 12165, 12166, 12167, 12168, 12169, 12170, 12171, 12172, 12173, 12174, 12175,
]

# subcortical lh/rh
REGION_CODES = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58]


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def region_average(sal, subject, region_code):
    mask_path = os.path.join(FREESURFER_DIR, str(subject), f"{region_code}.mgz")
    if not os.path.exists(mask_path):
        return None, 0
    mask = nib.load(mask_path).get_fdata()
    mask_bool = mask == 1.0
    n_vox = int(mask_bool.sum())
    if n_vox == 0:
        return None, 0
    return float(sal[mask_bool].mean()), n_vox


def main():
    manifest = load_manifest(MANIFEST_PATH)
    print(f"Loaded manifest: {len(manifest)} subjects, {len(REGION_CODES)} regions each")

    rows = []
    for row in manifest:
        subject = row["subject"]
        sal = np.load(row["saliency_path"])

        n_missing = 0
        for region_code in REGION_CODES:
            avg, n_vox = region_average(sal, subject, region_code)
            if avg is None:
                n_missing += 1
            rows.append({
                "subject": subject,
                "fold": row["fold"],
                "region_id": region_code,
                "avg_saliency": avg,
                "n_voxels": n_vox,
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "misclassified": row["misclassified"],
            })

        flag = f"  ({n_missing} missing/empty masks)" if n_missing else ""
        print(f"  {subject}: {len(REGION_CODES) - n_missing}/{len(REGION_CODES)} regions{flag}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows ({len(manifest)} subjects x {len(REGION_CODES)} regions) to {OUT_PATH}")

    subject_order = [row["subject"] for row in manifest]  
    pivot = {region_code: {} for region_code in REGION_CODES}
    for r in rows:
        pivot[r["region_id"]][r["subject"]] = r["avg_saliency"]

    with open(WIDE_OUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["AREA"] + subject_order)
        for region_code in REGION_CODES:
            row_vals = [pivot[region_code].get(sub, "") for sub in subject_order]
            row_vals = ["" if v is None else v for v in row_vals]
            writer.writerow([region_code] + row_vals)
    print(f"Wrote {len(REGION_CODES)} x {len(subject_order)} wide table to {WIDE_OUT_PATH}")


if __name__ == "__main__":
    main()
