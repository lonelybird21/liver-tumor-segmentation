"""Segmentation metrics that are meaningful for liver / tumour CT segmentation.

Overlap   : Dice, IoU (Jaccard), sensitivity, precision, specificity
Accuracy  : voxel accuracy -- reported for completeness, but it is dominated by
            background (~90-95% of a CT) and is NOT a quality measure here
Distance  : HD95 and average symmetric surface distance (ASSD), in mm
Volume    : volumes (mL) and relative volume difference (RVD)
Lesions   : per-lesion detection sensitivity / precision / F1 (connected components)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import ndimage as ndi

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(mask.ndim, 1), border_value=0)


def surface_distances(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]):
    """Distances (mm) from each surface voxel of one mask to the surface of the other."""
    sp, sg = _surface(pred), _surface(gt)
    # crop to the joint bounding box (+1) to keep the distance transforms cheap
    both = pred | gt
    idx = [np.where(both.any(axis=tuple(a for a in range(both.ndim) if a != ax)))[0] for ax in range(both.ndim)]
    sl = tuple(slice(max(i[0] - 1, 0), i[-1] + 2) for i in idx)
    sp, sg = sp[sl], sg[sl]
    dt_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return dt_g[sp], dt_p[sg]


def binary_metrics(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], distances: bool = True) -> Dict[str, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    tn = float(pred.size) - tp - fp - fn
    voxel_ml = float(np.prod(spacing)) / 1000.0
    out = {
        "gt_present": float(gt.any()),
        "pred_present": float(pred.any()),
        "dice": (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0,
        "iou": (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "voxel_accuracy": (tp + tn) / pred.size,
        "volume_pred_ml": (tp + fp) * voxel_ml,
        "volume_gt_ml": (tp + fn) * voxel_ml,
        "rvd": ((tp + fp) - (tp + fn)) / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "hd95_mm": float("nan"),
        "assd_mm": float("nan"),
        "tp": tp, "fp": fp, "fn": fn,
    }
    if distances and pred.any() and gt.any():
        d_pg, d_gp = surface_distances(pred, gt, spacing)
        alld = np.concatenate([d_pg, d_gp])
        out["hd95_mm"] = float(np.percentile(alld, 95))
        out["assd_mm"] = float(alld.mean())
    return out


def lesion_detection(pred_tumor: np.ndarray, gt_tumor: np.ndarray, spacing: Sequence[float], min_overlap: float = 0.0) -> Dict[str, float]:
    """A GT lesion counts as detected if a predicted component overlaps it
    (Dice with the union of overlapping predictions > `min_overlap`)."""
    gl, ng = ndi.label(gt_tumor, structure=STRUCT_26)
    pl, npred = ndi.label(pred_tumor, structure=STRUCT_26)
    pred_sizes = np.bincount(pl.ravel(), minlength=npred + 1)
    detected, small_total, small_detected = 0, 0, 0
    voxel = float(np.prod(spacing))
    for i, sl in enumerate(ndi.find_objects(gl), start=1):
        g = gl[sl] == i
        hit_ids = np.unique(pl[sl][g])
        hit_ids = hit_ids[hit_ids > 0]
        ok = False
        if len(hit_ids):
            inter = np.count_nonzero(g & np.isin(pl[sl], hit_ids))
            dice = 2 * inter / (pred_sizes[hit_ids].sum() + np.count_nonzero(g))
            ok = dice > min_overlap
        detected += ok
        diameter = 2 * (3 * np.count_nonzero(g) * voxel / (4 * np.pi)) ** (1 / 3)
        if diameter < 10.0:
            small_total += 1
            small_detected += ok
    tp_pred = 0
    for j, sl in enumerate(ndi.find_objects(pl), start=1):
        if np.any(gt_tumor[sl][pl[sl] == j]):
            tp_pred += 1
    sens = detected / ng if ng else float("nan")
    prec = tp_pred / npred if npred else float("nan")
    f1 = 2 * sens * prec / (sens + prec) if ng and npred and (sens + prec) > 0 else float("nan")
    return {
        "lesions_gt": float(ng),
        "lesions_pred": float(npred),
        "lesions_detected": float(detected),
        "lesion_sensitivity": sens,
        "lesion_precision": prec,
        "lesion_f1": f1,
        "false_positive_lesions": float(npred - tp_pred),
        "small_lesions_gt": float(small_total),
        "small_lesion_sensitivity": small_detected / small_total if small_total else float("nan"),
    }


def evaluate_case(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], distances: bool = True) -> Dict[str, float]:
    """pred/gt label maps {0,1,2}. Liver is evaluated as liver+tumour (LiTS convention)."""
    res: Dict[str, float] = {}
    for name, p, g in (
        ("liver", pred > 0, gt > 0),
        ("tumor", pred == 2, gt == 2),
    ):
        for k, v in binary_metrics(p, g, spacing, distances).items():
            res[f"{name}_{k}"] = v
    res.update({f"tumor_{k}": v for k, v in lesion_detection(pred == 2, gt == 2, spacing).items()})
    res["tumor_burden_pct_pred"] = 100 * res["tumor_volume_pred_ml"] / res["liver_volume_pred_ml"] if res["liver_volume_pred_ml"] else 0.0
    res["tumor_burden_pct_gt"] = 100 * res["tumor_volume_gt_ml"] / res["liver_volume_gt_ml"] if res["liver_volume_gt_ml"] else 0.0
    res["multiclass_voxel_accuracy"] = float(np.mean(pred == gt))
    return res


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def global_dice(rows, prefix: str) -> Optional[float]:
    """LiTS 'Dice global': pooled TP/FP/FN over all cases."""
    tp = sum(r[f"{prefix}_tp"] for r in rows)
    fp = sum(r[f"{prefix}_fp"] for r in rows)
    fn = sum(r[f"{prefix}_fn"] for r in rows)
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else None
