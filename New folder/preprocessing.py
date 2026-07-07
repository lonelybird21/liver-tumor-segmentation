"""
Preprocessing pipeline for Liver Tumor Segmentation (LiTS / MSD Task03).

Steps:
1. Load NIfTI volume + mask
2. Apply HU windowing (liver window)
3. Resample to consistent voxel spacing
4. Normalize intensities
5. Split by PATIENT (not by slice) into train/val/test
"""

import os
import glob
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import random

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
RAW_DIR = "data/raw"                 # expects: volume-XX.nii.gz, segmentation-XX.nii.gz
OUTPUT_DIR = "data"
TARGET_SPACING = (1.0, 1.0, 1.5)     # (x, y, z) mm - adjust based on your dataset stats
LIVER_WINDOW_LEVEL = 50              # HU window level for liver
LIVER_WINDOW_WIDTH = 400             # HU window width for liver

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def apply_windowing(volume, level=LIVER_WINDOW_LEVEL, width=LIVER_WINDOW_WIDTH):
    """Clip HU values to liver window and normalize to [0, 1]."""
    lower = level - width / 2
    upper = level + width / 2
    volume = np.clip(volume, lower, upper)
    volume = (volume - lower) / (upper - lower)
    return volume.astype(np.float32)


def resample_volume(volume, original_spacing, target_spacing=TARGET_SPACING, is_mask=False):
    """Resample volume to target voxel spacing."""
    resize_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(np.array(volume.shape) * resize_factor).astype(int)
    real_resize_factor = new_shape / np.array(volume.shape)

    order = 0 if is_mask else 1  # nearest-neighbor for masks, linear for images
    resampled = zoom(volume, real_resize_factor, order=order)

    if is_mask:
        resampled = np.round(resampled).astype(np.uint8)

    return resampled


def get_patient_ids(raw_dir=RAW_DIR):
    """Extract unique patient IDs from filenames like volume-12.nii.gz."""
    volume_files = glob.glob(os.path.join(raw_dir, "volume-*.nii.gz"))
    patient_ids = sorted(
        int(os.path.basename(f).replace("volume-", "").replace(".nii.gz", ""))
        for f in volume_files
    )
    return patient_ids


def split_patients(patient_ids, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """Split patient IDs into train/val/test. Ensures no patient leakage."""
    ids = patient_ids.copy()
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def process_patient(patient_id, raw_dir=RAW_DIR):
    """Load, window, resample, and normalize one patient's volume + mask."""
    vol_path = os.path.join(raw_dir, f"volume-{patient_id}.nii.gz")
    mask_path = os.path.join(raw_dir, f"segmentation-{patient_id}.nii.gz")

    vol_nii = nib.load(vol_path)
    mask_nii = nib.load(mask_path)

    volume = vol_nii.get_fdata()
    mask = mask_nii.get_fdata()

    original_spacing = vol_nii.header.get_zooms()[:3]

    # Windowing (only on the image, not the mask)
    volume = apply_windowing(volume)

    # Resample both volume and mask
    volume = resample_volume(volume, original_spacing, is_mask=False)
    mask = resample_volume(mask, original_spacing, is_mask=True)

    # Sanity check: shapes must match after resampling
    assert volume.shape == mask.shape, (
        f"Shape mismatch for patient {patient_id}: "
        f"volume {volume.shape} vs mask {mask.shape}"
    )

    return volume, mask


def save_split(patient_ids, split_name, raw_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    """Process and save all patients belonging to one split."""
    img_dir = os.path.join(output_dir, split_name, "images")
    mask_dir = os.path.join(output_dir, split_name, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for pid in patient_ids:
        try:
            volume, mask = process_patient(pid, raw_dir)
        except FileNotFoundError:
            print(f"[WARN] Missing file for patient {pid}, skipping.")
            continue
        except AssertionError as e:
            print(f"[WARN] {e}, skipping patient {pid}.")
            continue

        np.save(os.path.join(img_dir, f"patient_{pid}.npy"), volume)
        np.save(os.path.join(mask_dir, f"patient_{pid}.npy"), mask)
        print(f"[OK] Saved patient {pid} -> {split_name}")


def run_pipeline():
    patient_ids = get_patient_ids()
    print(f"Found {len(patient_ids)} patients.")

    train_ids, val_ids, test_ids = split_patients(patient_ids)
    print(f"Split -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    save_split(train_ids, "train")
    save_split(val_ids, "val")
    save_split(test_ids, "test")

    print("Preprocessing complete.")


if __name__ == "__main__":
    run_pipeline()
