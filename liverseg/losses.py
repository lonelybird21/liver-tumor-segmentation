"""Losses: soft Dice + (focal) cross-entropy, with deep-supervision weighting."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Batch soft Dice (computed over the whole batch, as in nnU-Net), background excluded by default."""

    def __init__(self, num_classes: int, include_background: bool = False, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, target):
        probs = logits.float().softmax(1)
        onehot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()
        dims = (0,) + tuple(range(2, logits.dim()))
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
        focal_gamma: float = 0.0,
        dice_include_background: bool = False,
    ):
        super().__init__()
        self.dice = SoftDiceLoss(num_classes, dice_include_background)
        self.dw, self.cw, self.gamma = dice_weight, ce_weight, focal_gamma
        self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32) if class_weights else None)

    def forward(self, logits, target):
        loss = 0.0
        if self.cw:
            ce = F.cross_entropy(logits.float(), target, weight=self.weight, reduction="none")
            if self.gamma > 0:
                pt = torch.exp(-F.cross_entropy(logits.float(), target, reduction="none"))
                ce = (1 - pt) ** self.gamma * ce
            loss = loss + self.cw * ce.mean()
        if self.dw:
            loss = loss + self.dw * self.dice(logits, target)
        return loss


class DeepSupervisionLoss(nn.Module):
    """Weights 1, 1/2, 1/4, ... (lowest resolution dropped), normalised to sum to 1."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, outputs: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor):
        if torch.is_tensor(outputs):
            return self.base(outputs, target)
        n = len(outputs)
        w = torch.tensor([0.5**i for i in range(n)])
        if n > 1:
            w[-1] = 0
        w = w / w.sum()
        total = 0.0
        for wi, out in zip(w.tolist(), outputs):
            if wi == 0:
                continue
            t = target
            if out.shape[2:] != target.shape[1:]:
                t = F.interpolate(target[:, None].float(), size=out.shape[2:], mode="nearest")[:, 0].long()
            total = total + wi * self.base(out, t)
        return total


def build_loss(cfg: dict) -> nn.Module:
    lc = cfg["train"]["loss"]
    base = DiceCELoss(
        num_classes=int(cfg["data"]["num_classes"]),
        dice_weight=lc["dice_weight"],
        ce_weight=lc["ce_weight"],
        class_weights=lc["class_weights"],
        focal_gamma=lc["focal_gamma"],
        dice_include_background=lc["dice_include_background"],
    )
    return DeepSupervisionLoss(base)
