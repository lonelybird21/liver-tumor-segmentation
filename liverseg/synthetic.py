"""Synthetic abdominal CT phantoms in LiTS file layout.

Only for smoke tests, CI and demos: verifies that preprocessing, training,
inference, orientation handling and evaluation all work end to end without
downloading real data. Numbers obtained on phantoms say nothing about real CT.

    liverseg-synthetic --out data/synthetic --n 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

ORIENTATIONS = ["LPS", "RAS", "LPI", "RAI"]


def make_case(rng: np.random.Generator, shape=(40, 96, 96), spacing=(2.5, 1.5, 1.5), n_tumors=None):
    """Return (hu, label) arrays in (z, y, x) LPS layout and spacing (z, y, x)."""
    nz, ny, nx = shape
    sz, sy, sx = spacing
    z, y, x = np.meshgrid(np.arange(nz) * sz, np.arange(ny) * sy, np.arange(nx) * sx, indexing="ij")
    cz, cy, cx = nz * sz / 2, ny * sy / 2, nx * sx / 2
    hu = np.full(shape, -1000.0, dtype=np.float32)
    body = ((y - cy) / (0.42 * ny * sy)) ** 2 + ((x - cx) / (0.47 * nx * sx)) ** 2 < 1
    hu[body] = -90.0  # fat
    inner = ((y - cy) / (0.36 * ny * sy)) ** 2 + ((x - cx) / (0.41 * nx * sx)) ** 2 < 1
    hu[inner] = 35.0  # soft tissue / bowel
    spine = ((y - (cy + 0.25 * ny * sy)) ** 2 + (x - cx) ** 2) < (0.07 * nx * sx) ** 2
    hu[spine] = 450.0
    # liver: patient right = low x in LPS, anterior = low y
    lz = cz + rng.uniform(-0.1, 0.1) * nz * sz
    ly = cy - rng.uniform(0.0, 0.08) * ny * sy
    lx = cx - rng.uniform(0.15, 0.2) * nx * sx
    rz, ry, rx = rng.uniform(0.28, 0.36) * nz * sz, rng.uniform(0.2, 0.25) * ny * sy, rng.uniform(0.17, 0.22) * nx * sx
    liver = ((z - lz) / rz) ** 2 + ((y - ly) / ry) ** 2 + ((x - lx) / rx) ** 2 < 1
    liver &= inner
    hu[liver] = rng.uniform(95, 115)
    # second organ with similar density (spleen) to force the model to use shape/position
    spl = ((z - lz) / (0.5 * rz)) ** 2 + ((y - cy) / (0.5 * ry)) ** 2 + ((x - (cx + 0.25 * nx * sx)) / (0.4 * rx)) ** 2 < 1
    hu[spl & inner & ~liver] = rng.uniform(95, 115)
    label = liver.astype(np.uint8)
    n_tumors = rng.integers(0, 4) if n_tumors is None else n_tumors
    for _ in range(n_tumors):
        pts = np.argwhere(liver)
        c = pts[rng.integers(len(pts))] * np.array(spacing)
        r = rng.uniform(4, 11)
        t = ((z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 < r**2) & liver
        hu[t] = rng.uniform(30, 55)
        label[t] = 2
    hu = gaussian_filter(hu, 0.7) + rng.normal(0, 12, shape).astype(np.float32)
    return hu.astype(np.float32), label


def to_sitk(arr_zyx, spacing_zyx, orientation="LPS"):
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing(tuple(float(s) for s in spacing_zyx[::-1]))
    img.SetOrigin((-100.0, -120.0, 300.0))
    return sitk.DICOMOrient(img, orientation) if orientation != "LPS" else img


def write_dataset(out_dir: str | Path, n: int = 12, seed: int = 0, shape=(40, 96, 96)):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        sp = (float(rng.choice([2.0, 2.5, 3.0])), float(rng.uniform(1.3, 1.7)), 0.0)
        sp = (sp[0], sp[1], sp[1])
        hu, lab = make_case(rng, shape, sp, n_tumors=1 + i % 3)
        ori = ORIENTATIONS[i % len(ORIENTATIONS)]
        img, seg = to_sitk(hu, sp, ori), to_sitk(lab, sp, ori)
        sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16), str(out / f"volume-{i}.nii.gz"))
        sitk.WriteImage(sitk.Cast(seg, sitk.sitkUInt8), str(out / f"segmentation-{i}.nii.gz"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(f"Wrote synthetic dataset to {write_dataset(a.out, a.n, a.seed)}")


if __name__ == "__main__":
    main()
