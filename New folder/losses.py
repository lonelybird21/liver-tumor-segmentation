"""
Combined Dice + Cross-Entropy loss.
Handles severe class imbalance between background / liver / tumor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits:  (B, C, D, H, W) raw model output
        targets: (B, D, H, W) integer labels {0, 1, 2}
        """
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality = torch.sum(probs + targets_onehot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Average over classes (you can also weight tumor class higher here)
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """Dice + weighted Cross-Entropy. Tumor class gets higher weight
    because it is heavily underrepresented compared to background/liver."""

    def __init__(self, class_weights=(0.2, 0.3, 0.5), num_classes=3):
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + dice_loss


def dice_score_per_class(logits, targets, num_classes=3, smooth=1e-5):
    """Utility for evaluation: returns Dice score per class (not loss, the actual metric)."""
    preds = torch.argmax(logits, dim=1)
    scores = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        scores.append(dice.item())
    return scores  # [dice_background, dice_liver, dice_tumor]
