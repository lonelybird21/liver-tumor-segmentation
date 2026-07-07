"""
Training script for Liver Tumor Segmentation (U-Net 3D).
Includes checkpointing and early stopping since 3D training can take days.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import UNet3D
from losses import CombinedLoss, dice_score_per_class

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
DATA_DIR = "data"
CHECKPOINT_DIR = "outputs/checkpoints"
PATCH_SIZE = (64, 128, 128)
BATCH_SIZE = 2
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class LiverPatchDataset(Dataset):
    """Loads preprocessed .npy volumes and extracts random patches for training."""

    def __init__(self, split, data_dir=DATA_DIR, patch_size=PATCH_SIZE, train=True):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.mask_dir = os.path.join(data_dir, split, "masks")
        self.patients = sorted(os.listdir(self.img_dir))
        self.patch_size = patch_size
        self.train = train

    def __len__(self):
        return len(self.patients)

    def _random_patch(self, volume, mask):
        d, h, w = volume.shape
        pd, ph, pw = self.patch_size

        d0 = np.random.randint(0, max(1, d - pd))
        h0 = np.random.randint(0, max(1, h - ph))
        w0 = np.random.randint(0, max(1, w - pw))

        vol_patch = volume[d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]
        mask_patch = mask[d0:d0 + pd, h0:h0 + ph, w0:w0 + pw]

        # Pad if volume smaller than patch size
        pad_d = pd - vol_patch.shape[0]
        pad_h = ph - vol_patch.shape[1]
        pad_w = pw - vol_patch.shape[2]
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            vol_patch = np.pad(vol_patch, ((0, pad_d), (0, pad_h), (0, pad_w)))
            mask_patch = np.pad(mask_patch, ((0, pad_d), (0, pad_h), (0, pad_w)))

        return vol_patch, mask_patch

    def __getitem__(self, idx):
        fname = self.patients[idx]
        volume = np.load(os.path.join(self.img_dir, fname))
        mask = np.load(os.path.join(self.mask_dir, fname))

        vol_patch, mask_patch = self._random_patch(volume, mask)

        vol_tensor = torch.from_numpy(vol_patch).float().unsqueeze(0)  # (1, D, H, W)
        mask_tensor = torch.from_numpy(mask_patch).long()              # (D, H, W)

        return vol_tensor, mask_tensor


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for volumes, masks in loader:
        volumes, masks = volumes.to(device), masks.to(device)

        optimizer.zero_grad()
        logits = model(volumes)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []  # list of [bg, liver, tumor]

    with torch.no_grad():
        for volumes, masks in loader:
            volumes, masks = volumes.to(device), masks.to(device)
            logits = model(volumes)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            dice_scores.append(dice_score_per_class(logits, masks))

    avg_loss = total_loss / len(loader)
    avg_dice = np.mean(dice_scores, axis=0)  # [dice_bg, dice_liver, dice_tumor]
    return avg_loss, avg_dice


def main():
    train_dataset = LiverPatchDataset("train", train=True)
    val_dataset = LiverPatchDataset("val", train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = UNet3D(in_channels=1, num_classes=3, base_filters=16).to(DEVICE)
    criterion = CombinedLoss(class_weights=(0.2, 0.3, 0.5), num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice = validate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Dice[bg={val_dice[0]:.3f}, liver={val_dice[1]:.3f}, tumor={val_dice[2]:.3f}]"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  -> Saved new best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break


if __name__ == "__main__":
    main()
