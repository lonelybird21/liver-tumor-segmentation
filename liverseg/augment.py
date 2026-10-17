"""Data augmentation for 2D and 3D patches (nnU-Net-style set of transforms).

Image tensors are (C, *spatial), label tensors are (*spatial); spatial order is
(z, y, x) for 3D and (y, x) for 2D.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def _rotation_zyx(ax: float, ay: float, az: float) -> np.ndarray:
    """Rotation matrix acting on (z, y, x) coordinates; az rotates the axial (y-x) plane."""
    cz, sz = math.cos(az), math.sin(az)
    cy, sy = math.cos(ay), math.sin(ay)
    cx, sx = math.cos(ax), math.sin(ax)
    rz = np.array([[1, 0, 0], [0, cz, -sz], [0, sz, cz]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])
    return rz @ ry @ rx


def spatial_transform(
    image: torch.Tensor,
    label: torch.Tensor,
    out_size: Sequence[int],
    angles_rad: Sequence[float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
):
    """Rotate/scale about the crop centre and resample to `out_size`.

    `image` is (C, *in_size) and should be larger than `out_size` so rotated
    corners stay inside real data. Scale > 1 zooms in.
    """
    nd = image.dim() - 1
    in_size = np.array(image.shape[1:], dtype=np.float64)
    out = np.array(out_size, dtype=np.float64)
    if nd == 3:
        rot = _rotation_zyx(*angles_rad)
    else:
        a = angles_rad[-1]
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    m_zyx = np.diag(2.0 / in_size) @ rot @ np.diag(out / 2.0) / scale
    m_xyz = m_zyx[::-1, ::-1].copy()  # torch grids are ordered (x, y[, z])
    theta = torch.zeros(1, nd, nd + 1, dtype=torch.float32)
    theta[0, :, :nd] = torch.from_numpy(m_xyz).float()
    size = (1, 1, *[int(s) for s in out_size])
    grid = F.affine_grid(theta, size, align_corners=False)

    pad = image.amin(dim=tuple(range(1, nd + 1)), keepdim=True)
    img = F.grid_sample((image - pad)[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0] + pad
    lab = F.grid_sample(label[None, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
    return img, lab.round().to(label.dtype)


def gaussian_noise(img, rng, std_range):
    std = rng.uniform(*std_range)
    return img + torch.randn_like(img) * std


def gaussian_blur(img, rng, sigma_range):
    out = img.numpy().copy()
    for c in range(out.shape[0]):
        out[c] = gaussian_filter(out[c], rng.uniform(*sigma_range))
    return torch.from_numpy(out)


def brightness(img, rng, rng_range):
    return img * rng.uniform(*rng_range)


def contrast(img, rng, rng_range):
    lo, hi = img.min(), img.max()
    mean = img.mean()
    return ((img - mean) * rng.uniform(*rng_range) + mean).clamp(lo, hi)


def simulate_low_resolution(img, rng, zoom_range):
    nd = img.dim() - 1
    zoom = rng.uniform(*zoom_range)
    shape = img.shape[1:]
    small = [max(1, int(round(s * zoom))) for s in shape]
    x = F.interpolate(img[None], size=small, mode="nearest")
    mode = "trilinear" if nd == 3 else "bilinear"
    return F.interpolate(x, size=list(shape), mode=mode, align_corners=False)[0]


def gamma(img, rng, gamma_range):
    lo, hi = img.min(), img.max()
    rngv = (hi - lo).clamp_min(1e-6)
    g = rng.uniform(*gamma_range)
    return ((img - lo) / rngv).clamp(0, 1).pow(g) * rngv + lo


def mirror(img, lab, rng, axes, p):
    for ax in axes:
        if rng.random() < p:
            img = torch.flip(img, dims=(ax + 1,))
            lab = torch.flip(lab, dims=(ax,))
    return img, lab


def intensity_augment(img: torch.Tensor, rng: np.random.Generator, a: dict) -> torch.Tensor:
    if rng.random() < a["p_noise"]:
        img = gaussian_noise(img, rng, a["noise_std"])
    if rng.random() < a["p_blur"]:
        img = gaussian_blur(img, rng, a["blur_sigma"])
    if rng.random() < a["p_brightness"]:
        img = brightness(img, rng, a["brightness_range"])
    if rng.random() < a["p_contrast"]:
        img = contrast(img, rng, a["contrast_range"])
    if rng.random() < a["p_lowres"]:
        img = simulate_low_resolution(img, rng, a["lowres_range"])
    if rng.random() < a["p_gamma"]:
        img = gamma(img, rng, a["gamma_range"])
    return img
