"""Anatomically-informed post-processing of a (z, y, x) label map {0 bg, 1 liver, 2 tumour}."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask, structure=STRUCT_26)
    if n <= 1:
        return mask.astype(bool)
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def postprocess(
    label: np.ndarray,
    spacing: Sequence[float],
    liver_largest_cc: bool = True,
    tumor_inside_liver: bool = True,
    fill_liver_holes: bool = True,
    min_tumor_volume_mm3: float = 10.0,
) -> np.ndarray:
    out = label.astype(np.uint8).copy()
    organ = out > 0
    if liver_largest_cc and organ.any():
        keep = largest_component(organ)
        if tumor_inside_liver:
            out[~keep] = 0
        else:
            out[(~keep) & (out == 1)] = 0
        organ = out > 0
    if fill_liver_holes and organ.any():
        filled = np.zeros_like(organ)
        for z in range(organ.shape[0]):  # slice-wise fill is robust for open 3D shapes
            filled[z] = ndi.binary_fill_holes(organ[z])
        out[filled & (out == 0)] = 1
    if min_tumor_volume_mm3 > 0 and (out == 2).any():
        voxel = float(np.prod(spacing))
        lab, n = ndi.label(out == 2, structure=STRUCT_26)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:] * voxel
        small = np.flatnonzero(sizes < min_tumor_volume_mm3) + 1
        if len(small):
            out[np.isin(lab, small)] = 1
    return out
