"""Quantitative report from a segmentation: liver volume, tumour burden, per-lesion
size table, and overlay figures (radiological display convention)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)
DISCLAIMER = (
    "Research software, not a medical device. Automated measurements must be reviewed "
    "by a qualified radiologist before any clinical use."
)


def _max_pairwise(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    if len(points) > 3:
        try:
            points = points[ConvexHull(points).vertices]
        except (QhullError, ValueError):
            pass
    d = points[:, None, :] - points[None, :, :]
    return float(np.sqrt((d**2).sum(-1)).max())


def longest_axial_diameter(mask: np.ndarray, spacing: Sequence[float]) -> float:
    """Longest in-plane diameter (mm) over all axial slices (RECIST-style long axis)."""
    best = 0.0
    sy, sx = spacing[1], spacing[2]
    for z in np.where(mask.any(axis=(1, 2)))[0]:
        pts = np.argwhere(mask[z]).astype(np.float64)
        # use pixel corners so a 1-pixel lesion still has a size
        corners = np.concatenate([pts + [dy, dx] for dy in (0, 1) for dx in (0, 1)]) * [sy, sx]
        best = max(best, _max_pairwise(np.unique(corners, axis=0)))
    return best


def lesion_table(label: np.ndarray, spacing: Sequence[float]) -> List[Dict]:
    lab, n = ndi.label(label == 2, structure=STRUCT_26)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        vol = m.sum() * voxel_ml
        com = np.array(ndi.center_of_mass(m)) + [s.start for s in sl]
        rows.append({
            "lesion": i,
            "volume_ml": round(float(vol), 3),
            "longest_axial_diameter_mm": round(longest_axial_diameter(m, spacing), 1),
            "equivalent_sphere_diameter_mm": round(float(2 * (3 * vol * 1000 / (4 * np.pi)) ** (1 / 3)), 1),
            "slice_range": [int(sl[0].start), int(sl[0].stop - 1)],
            "centroid_zyx_voxel": [int(round(c)) for c in com],
        })
    rows.sort(key=lambda r: -r["volume_ml"])
    for k, r in enumerate(rows, start=1):
        r["lesion"] = k
    return rows


def summarize(label: np.ndarray, spacing: Sequence[float]) -> Dict:
    voxel_ml = float(np.prod(spacing)) / 1000.0
    liver_total = float((label > 0).sum() * voxel_ml)
    tumor = float((label == 2).sum() * voxel_ml)
    lesions = lesion_table(label, spacing)
    return {
        "liver_volume_ml": round(liver_total, 1),
        "liver_parenchyma_volume_ml": round(liver_total - tumor, 1),
        "tumor_volume_ml": round(tumor, 2),
        "tumor_burden_percent": round(100 * tumor / liver_total, 2) if liver_total else 0.0,
        "number_of_lesions": len(lesions),
        "lesions_over_10mm": sum(1 for l in lesions if l["longest_axial_diameter_mm"] >= 10),
        "lesions": lesions,
        "voxel_spacing_zyx_mm": [round(float(s), 3) for s in spacing],
        "disclaimer": DISCLAIMER,
    }


def render_overlays(hu: np.ndarray, label: np.ndarray, spacing: Sequence[float], path: str | Path, level=50, width=400):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    lo, hi = level - width / 2, level + width / 2
    cmap = ListedColormap([(0, 0, 0, 0), (0.1, 0.85, 0.3, 0.35), (1.0, 0.15, 0.15, 0.6)])
    picks = []
    if (label == 2).any():
        picks.append(("largest tumour slice", int(np.argmax((label == 2).sum(axis=(1, 2))))))
    if (label > 0).any():
        picks.append(("largest liver slice", int(np.argmax((label > 0).sum(axis=(1, 2))))))
    if not picks:
        picks.append(("middle slice", label.shape[0] // 2))
    show_cor = label.shape[0] > 4
    ncols = len(picks) + int(show_cor)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)
    for ax, (title, z) in zip(axes[0], picks):
        ax.imshow(np.clip(hu[z], lo, hi), cmap="gray", vmin=lo, vmax=hi, aspect=spacing[1] / spacing[2])
        ax.imshow(label[z], cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect=spacing[1] / spacing[2])
        if (label[z] > 0).any():
            ax.contour(label[z] > 0, levels=[0.5], colors=["lime"], linewidths=0.8)
        if (label[z] == 2).any():
            ax.contour(label[z] == 2, levels=[0.5], colors=["red"], linewidths=0.8)
        ax.set_title(f"axial {z} ({title})")
        ax.axis("off")
    if show_cor:
        ax = axes[0][-1]
        y = int(np.argmax((label > 0).sum(axis=(0, 2)))) if (label > 0).any() else label.shape[1] // 2
        ax.imshow(np.flipud(np.clip(hu[:, y], lo, hi)), cmap="gray", vmin=lo, vmax=hi, aspect=spacing[0] / spacing[2])
        ax.imshow(np.flipud(label[:, y]), cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect=spacing[0] / spacing[2])
        ax.set_title(f"coronal {y}")
        ax.axis("off")
    fig.suptitle("green = liver, red = tumour  |  patient right on image left", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(hu: np.ndarray, label: np.ndarray, spacing: Sequence[float], out_dir: str | Path, name: str) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(label, spacing)
    s["case"] = name
    (out / f"{name}_report.json").write_text(json.dumps(s, indent=2))
    render_overlays(hu, label, spacing, out / f"{name}_overlay.png")
    lines = [
        f"# Liver segmentation report: {name}",
        "",
        f"- Liver volume (incl. tumour): **{s['liver_volume_ml']} mL**",
        f"- Liver parenchyma volume: {s['liver_parenchyma_volume_ml']} mL",
        f"- Total tumour volume: **{s['tumor_volume_ml']} mL** ({s['tumor_burden_percent']}% of liver)",
        f"- Lesions: **{s['number_of_lesions']}** ({s['lesions_over_10mm']} with long axis >= 10 mm)",
        "",
    ]
    if s["lesions"]:
        lines += ["| # | Volume (mL) | Long axis (mm) | Eq. diameter (mm) | Slices |", "|---|---|---|---|---|"]
        for l in s["lesions"]:
            lines.append(
                f"| {l['lesion']} | {l['volume_ml']} | {l['longest_axial_diameter_mm']} | "
                f"{l['equivalent_sphere_diameter_mm']} | {l['slice_range'][0]}-{l['slice_range'][1]} |"
            )
        lines.append("")
    lines += [f"![overlay]({name}_overlay.png)", "", f"_{DISCLAIMER}_", ""]
    (out / f"{name}_report.md").write_text("\n".join(lines))
    return s
