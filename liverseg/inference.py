"""Inference: Gaussian-weighted sliding window, mirror test-time augmentation,
fold ensembling, optional two-stage cascade (coarse liver locator -> fine model),
post-processing, and mapping the result back onto the original scan grid."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from .io import label_to_original_grid, orient, to_array
from .models import build_model
from .postprocess import postprocess
from .preprocessing import bbox_of, effective_spacing, expand_bbox, normalize_hu, resample_array


# --------------------------------------------------------------------------- #
# sliding window
# --------------------------------------------------------------------------- #
def gaussian_importance(patch: Sequence[int], sigma_scale: float = 1 / 8) -> torch.Tensor:
    grids = torch.meshgrid(*[torch.arange(p, dtype=torch.float32) for p in patch], indexing="ij")
    g = torch.zeros(patch, dtype=torch.float32)
    for ax, p in enumerate(patch):
        g = g + ((grids[ax] - (p - 1) / 2) ** 2) / (2 * (p * sigma_scale) ** 2)
    g = torch.exp(-g)
    g = g / g.max()
    return g.clamp_min(g[g > 0].min())


def window_starts(size: int, patch: int, overlap: float) -> List[int]:
    if size <= patch:
        return [0]
    step = max(1, int(patch * (1 - overlap)))
    n = int(np.ceil((size - patch) / step)) + 1
    return sorted({int(round(i * (size - patch) / (n - 1))) for i in range(n)})


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: Sequence[int],
    num_classes: int,
    overlap: float = 0.5,
    mirror_axes: Sequence[int] = (),
    batch_size: int = 2,
    device: str | torch.device = "cpu",
    amp: bool = True,
) -> np.ndarray:
    """image: (C, *spatial) normalised array -> softmax probabilities (K, *spatial)."""
    device = torch.device(device)
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    spatial = list(x.shape[1:])
    pad = []
    for s, p in zip(spatial[::-1], list(patch_size)[::-1]):
        total = max(0, p - s)
        pad += [total // 2, total - total // 2]
    if any(pad):
        fill = float(x.min())
        x = F.pad(x[None], pad, mode="constant", value=fill)[0]
    padded = list(x.shape[1:])

    starts = [window_starts(s, p, overlap) for s, p in zip(padded, patch_size)]
    positions = list(itertools.product(*starts))
    gauss = gaussian_importance(patch_size).to(device)
    acc = torch.zeros((num_classes, *padded), dtype=torch.float32)
    norm = torch.zeros(padded, dtype=torch.float32)
    flips = [c for r in range(len(mirror_axes) + 1) for c in itertools.combinations(mirror_axes, r)]
    use_amp = amp and device.type == "cuda"

    for i in range(0, len(positions), batch_size):
        batch_pos = positions[i : i + batch_size]
        sl = [tuple(slice(st, st + p) for st, p in zip(pos, patch_size)) for pos in batch_pos]
        xb = torch.stack([x[(slice(None),) + s] for s in sl]).to(device)
        prob = 0
        for flip in flips:
            dims = tuple(a + 2 for a in flip)
            inp = torch.flip(xb, dims) if dims else xb
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(inp)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.float().softmax(1)
            prob = prob + (torch.flip(out, dims) if dims else out)
        prob = (prob / len(flips)) * gauss
        prob = prob.cpu()
        g = gauss.cpu()
        for j, s in enumerate(sl):
            acc[(slice(None),) + s] += prob[j]
            norm[s] += g

    acc /= norm.clamp_min(1e-8)
    if any(pad):
        crop = []
        # F.pad lists (before, after) from the last axis backwards; flip to axis order
        pairs = [(pad[2 * k], pad[2 * k + 1]) for k in range(len(spatial))][::-1]
        for (before, _), s in zip(pairs, spatial):
            crop.append(slice(before, before + s))
        acc = acc[(slice(None),) + tuple(crop)]
    return acc.numpy()


def probs_to_label(probs: np.ndarray, out_shape: Sequence[int]) -> np.ndarray:
    """Resize class probabilities to `out_shape` (trilinear, class by class) and take argmax."""
    out_shape = tuple(int(s) for s in out_shape)
    if tuple(probs.shape[1:]) == out_shape:
        return probs.argmax(0).astype(np.uint8)
    best = np.full(out_shape, -1.0, dtype=np.float32)
    lab = np.zeros(out_shape, dtype=np.uint8)
    for c in range(probs.shape[0]):
        t = torch.from_numpy(np.ascontiguousarray(probs[c]))[None, None]
        v = F.interpolate(t, size=out_shape, mode="trilinear", align_corners=False)[0, 0].numpy()
        m = v > best
        best[m] = v[m]
        lab[m] = c
    return lab


# --------------------------------------------------------------------------- #
# predictor
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    label: sitk.Image                 # on the original image grid
    label_oriented: np.ndarray        # (z, y, x) in canonical LPS orientation
    image_oriented: np.ndarray        # HU, same grid as label_oriented
    spacing: tuple                    # (z, y, x) mm of the oriented grid


def load_checkpoint(path: str | Path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt["intensity"]


def find_checkpoints(path: str | Path, name: str = "checkpoint_best.pt") -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    found = sorted(p.glob(f"fold_*/{name}")) or sorted(p.glob(name)) or sorted(p.glob("**/*.pt"))
    if not found:
        raise FileNotFoundError(f"No checkpoints found in {p}")
    return found


class Predictor:
    def __init__(
        self,
        checkpoints: Sequence[str | Path],
        device: Optional[str] = None,
        tta: Optional[bool] = None,
        overlap: Optional[float] = None,
        locator: Optional["Predictor"] = None,
        do_postprocess: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.members = [load_checkpoint(c, self.device) for c in checkpoints]
        cfg0 = self.members[0][1]
        for _, c, _ in self.members[1:]:
            if c["model"]["dims"] != cfg0["model"]["dims"] or c["data"]["target_spacing"] != cfg0["data"]["target_spacing"]:
                raise ValueError("Ensembled checkpoints must share dims and target spacing")
        self.cfg = cfg0
        inf = cfg0["inference"]
        self.tta = inf["tta"] if tta is None else tta
        self.overlap = inf["overlap"] if overlap is None else overlap
        self.locator = locator
        self.do_postprocess = do_postprocess
        self.num_classes = int(cfg0["data"]["num_classes"])
        self.dims = int(cfg0["model"]["dims"])

    @classmethod
    def from_path(cls, path, **kw) -> "Predictor":
        return cls(find_checkpoints(path), **kw)

    # -- core ---------------------------------------------------------------
    def _mirror_axes(self, cfg):
        if not self.tta:
            return ()
        axes = cfg["train"]["augmentation"]["mirror_axes"]
        return tuple(axes) if self.dims == 3 else tuple(a - 1 for a in axes if a > 0)

    def _predict_resampled(self, x_norm: np.ndarray, model, cfg) -> np.ndarray:
        """x_norm: normalised (z, y, x) at model spacing -> probs (K, z, y, x)."""
        bs = cfg["inference"]["batch_size"]
        if self.dims == 3:
            return sliding_window_predict(
                model, x_norm[None], cfg["train"]["patch_size"], self.num_classes,
                self.overlap, self._mirror_axes(cfg), bs, self.device,
            )
        k = int(cfg["model"]["context_slices"])
        out = np.zeros((self.num_classes, *x_norm.shape), dtype=np.float32)
        for z in range(x_norm.shape[0]):
            idx = np.clip(np.arange(z - k, z + k + 1), 0, x_norm.shape[0] - 1)
            out[:, z] = sliding_window_predict(
                model, x_norm[idx], cfg["train"]["patch_size"][1:], self.num_classes,
                self.overlap, self._mirror_axes(cfg), bs, self.device,
            )
        return out

    def predict_probs(self, hu: np.ndarray, spacing: Sequence[float]):
        """HU (z, y, x) at `spacing` -> (mean probs at model spacing, model spacing)."""
        target = effective_spacing(spacing, self.cfg["data"]["target_spacing"])
        hu_r = resample_array(hu, spacing, target)
        acc = None
        for model, cfg, intensity in self.members:
            p = self._predict_resampled(normalize_hu(hu_r, intensity), model, cfg)
            acc = p if acc is None else acc + p
        return acc / len(self.members), target

    def predict_label_array(self, hu: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
        if self.dims == 3 and hu.shape[0] < 4:
            raise ValueError(
                f"Input has only {hu.shape[0]} slice(s). A 3D model needs a CT volume; "
                "for single slices or pictures, train/use the 2D model (configs/lits_2d.yaml)."
            )
        roi = None
        if self.locator is not None:
            coarse = self.locator.predict_label_array(hu, spacing)
            box = bbox_of(coarse > 0)
            if box is not None:
                margin = [int(np.ceil(m / s)) for m, s in zip(self.cfg["data"]["roi_margin_mm"], spacing)]
                roi = tuple(slice(lo, hi) for lo, hi in expand_bbox(box, margin, hu.shape))
        sub = hu[roi] if roi else hu
        probs, _ = self.predict_probs(sub, spacing)
        lab = probs_to_label(probs, sub.shape)
        if roi:
            full = np.zeros(hu.shape, dtype=np.uint8)
            full[roi] = lab
            lab = full
        if self.do_postprocess:
            lab = postprocess(lab, spacing, **self.cfg["inference"]["postprocess"])
        return lab

    def predict(self, image: sitk.Image) -> Prediction:
        img_o = orient(image)
        hu, spacing = to_array(img_o)
        lab = self.predict_label_array(hu.astype(np.float32), spacing)
        return Prediction(label_to_original_grid(lab, img_o, image), lab, hu, spacing)
