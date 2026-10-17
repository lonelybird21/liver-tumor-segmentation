"""Dataset preprocessing.

For every case:
  1. read CT + label, force the label onto the CT grid, reorient both to LPS
  2. resample to the target spacing (image: trilinear, label: one-hot linear + argmax)
  3. crop to the body (or, for the cascade's 2nd stage, to the liver + margin)
  4. store HU as int16 + label as uint8 (.npy, memory-mappable) and a metadata pickle
     with per-class voxel locations for foreground oversampling

Then: patient-level split (held-out test set + k cross-validation folds) and
intensity statistics computed from the *training* patients only (no test leakage).

Usage:
    python -m liverseg.preprocessing --config configs/lits_3d_fullres.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config
from .io import orient, read_image, read_label, to_array

HU_BODY_THRESHOLD = -500.0


# --------------------------------------------------------------------------- #
# array utilities shared with inference
# --------------------------------------------------------------------------- #
def effective_spacing(src: Sequence[float], target: Sequence[Optional[float]]) -> Tuple[float, float, float]:
    return tuple(float(s) if t is None else float(t) for s, t in zip(src, target))


def resampled_shape(shape: Sequence[int], src: Sequence[float], dst: Sequence[float]) -> Tuple[int, ...]:
    return tuple(max(1, int(round(n * s / d))) for n, s, d in zip(shape, src, dst))


def resize_array(arr: np.ndarray, new_shape: Sequence[int], is_label: bool = False, num_classes: int = 3) -> np.ndarray:
    """Resize a (z, y, x) array. Labels use per-class linear interpolation + argmax,
    which gives smoother, less aliased boundaries than nearest neighbour."""
    new_shape = tuple(int(s) for s in new_shape)
    if tuple(arr.shape) == new_shape:
        return arr.copy()
    if not is_label:
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))[None, None]
        return F.interpolate(t, size=new_shape, mode="trilinear", align_corners=False)[0, 0].numpy()
    best_val = np.full(new_shape, -1.0, dtype=np.float32)
    best_idx = np.zeros(new_shape, dtype=np.uint8)
    for c in range(num_classes):
        onehot = torch.from_numpy((arr == c).astype(np.float32))[None, None]
        val = F.interpolate(onehot, size=new_shape, mode="trilinear", align_corners=False)[0, 0].numpy()
        better = val > best_val
        best_val[better] = val[better]
        best_idx[better] = c
    return best_idx


def resample_array(arr, src_spacing, dst_spacing, is_label=False, num_classes=3) -> np.ndarray:
    return resize_array(arr, resampled_shape(arr.shape, src_spacing, dst_spacing), is_label, num_classes)


def normalize_hu(hu: np.ndarray, intensity: dict) -> np.ndarray:
    x = np.clip(hu.astype(np.float32), intensity["clip_low"], intensity["clip_high"])
    return (x - intensity["mean"]) / max(intensity["std"], 1e-6)


def bbox_of(mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], ...]]:
    if not mask.any():
        return None
    out = []
    for ax in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != ax)
        idx = np.where(mask.any(axis=other))[0]
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return tuple(out)


def expand_bbox(bbox, margin_vox, shape):
    return tuple((max(0, lo - m), min(n, hi + m)) for (lo, hi), m, n in zip(bbox, margin_vox, shape))


def class_locations(seg: np.ndarray, num_classes: int, rng: np.random.Generator, max_per_class: int = 20000) -> Dict[int, np.ndarray]:
    locs = {}
    for c in range(1, num_classes):
        coords = np.argwhere(seg == c)
        if len(coords) > max_per_class:
            coords = coords[rng.choice(len(coords), max_per_class, replace=False)]
        locs[c] = coords.astype(np.int32)
    return locs


# --------------------------------------------------------------------------- #
# dataset discovery & splitting
# --------------------------------------------------------------------------- #
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def find_cases(raw_dir: str | Path, fmt: str = "lits") -> List[Tuple[str, Path, Path]]:
    raw = Path(raw_dir)
    cases = []
    if fmt == "lits":
        for img in raw.rglob("volume-*.nii*"):
            cid = re.sub(r"\.nii(\.gz)?$", "", img.name).replace("volume-", "")
            seg = next(iter(sorted(raw.rglob(f"segmentation-{cid}.nii*"))), None)
            if seg is not None:
                cases.append((f"case_{cid}", img, seg))
    elif fmt in ("msd", "nnunet"):
        for img in (raw / "imagesTr").glob("*.nii*"):
            if img.name.startswith("."):
                continue
            stem = re.sub(r"\.nii(\.gz)?$", "", img.name)
            cid = stem[:-5] if fmt == "nnunet" and stem.endswith("_0000") else stem
            seg = next(iter(sorted((raw / "labelsTr").glob(f"{cid}.nii*"))), None)
            if seg is not None:
                cases.append((cid, img, seg))
    else:
        raise ValueError(f"Unknown data.format {fmt!r}")
    if not cases:
        raise FileNotFoundError(f"No labelled cases found in {raw} (format={fmt})")
    return sorted(cases, key=lambda c: _natural_key(c[0]))


def make_splits(case_ids: Sequence[str], num_folds: int, test_fraction: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    ids = list(case_ids)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_fraction))
    test, rest = sorted(ids[:n_test], key=_natural_key), ids[n_test:]
    folds = []
    if num_folds <= 1:
        n_val = max(1, int(round(0.2 * len(rest))))
        folds.append({"train": sorted(rest[n_val:], key=_natural_key), "val": sorted(rest[:n_val], key=_natural_key)})
    else:
        k = min(num_folds, len(rest))
        chunks = [rest[i::k] for i in range(k)]
        for i in range(k):
            val = chunks[i]
            train = [c for j, ch in enumerate(chunks) if j != i for c in ch]
            folds.append({"train": sorted(train, key=_natural_key), "val": sorted(val, key=_natural_key)})
    return {"seed": seed, "test": test, "folds": folds}


# --------------------------------------------------------------------------- #
# per-case processing
# --------------------------------------------------------------------------- #
def load_case_oriented(img_path, seg_path):
    img = read_image(img_path)
    seg = read_label(seg_path)
    if seg.GetSize() == img.GetSize():
        seg.CopyInformation(img)  # LiTS label headers sometimes differ by rounding
    else:
        raise ValueError(f"Image/label size mismatch: {img.GetSize()} vs {seg.GetSize()} ({img_path})")
    img_o, seg_o = orient(img), orient(seg)
    hu, spacing = to_array(img_o)
    lab, _ = to_array(seg_o)
    return hu.astype(np.float32), lab.astype(np.uint8), spacing


def preprocess_case(job) -> Optional[dict]:
    case_id, img_path, seg_path, data_cfg, out_dir = job
    num_classes = int(data_cfg["num_classes"])
    rng = np.random.default_rng(zlib.crc32(case_id.encode()))
    hu, lab, spacing = load_case_oriented(img_path, seg_path)
    lab[lab >= num_classes] = num_classes - 1

    fg_vals = hu[lab > 0]
    if len(fg_vals) > 20000:
        fg_vals = rng.choice(fg_vals, 20000, replace=False)

    target = effective_spacing(spacing, data_cfg["target_spacing"])
    hu_r = resample_array(hu, spacing, target, is_label=False)
    lab_r = resample_array(lab, spacing, target, is_label=True, num_classes=num_classes)

    if data_cfg.get("roi_crop"):
        box = bbox_of(lab_r > 0)
        if box is None:
            print(f"[skip] {case_id}: no liver in label, cannot build ROI crop")
            return None
        margin = [int(np.ceil(m / s)) for m, s in zip(data_cfg["roi_margin_mm"], target)]
    else:
        box = bbox_of(hu_r > HU_BODY_THRESHOLD) or tuple((0, n) for n in hu_r.shape)
        margin = [2, 5, 5]
    box = expand_bbox(box, margin, hu_r.shape)
    sl = tuple(slice(lo, hi) for lo, hi in box)
    hu_c, lab_c = hu_r[sl], lab_r[sl]

    out = Path(out_dir)
    np.save(out / f"{case_id}_img.npy", np.clip(np.round(hu_c), -32768, 32767).astype(np.int16))
    np.save(out / f"{case_id}_seg.npy", lab_c.astype(np.uint8))
    meta = {
        "case_id": case_id,
        "image_path": str(img_path),
        "label_path": str(seg_path),
        "original_spacing": spacing,
        "original_shape": tuple(hu.shape),
        "spacing": target,
        "shape": tuple(hu_c.shape),
        "crop_bbox": box,
        "class_locations": class_locations(lab_c, num_classes, rng),
        "voxels_per_class": {int(c): int((lab_c == c).sum()) for c in range(num_classes)},
    }
    with open(out / f"{case_id}.pkl", "wb") as f:
        pickle.dump(meta, f)
    return {"case_id": case_id, "fg_values": fg_vals.astype(np.float32), "shape": meta["shape"], "spacing": spacing}


def intensity_statistics(values: np.ndarray, clip_hu) -> dict:
    if isinstance(clip_hu, str) and clip_hu == "auto":
        lo, hi = np.percentile(values, [0.5, 99.5])
    else:
        lo, hi = clip_hu
    v = np.clip(values, lo, hi)
    return {"clip_low": float(lo), "clip_high": float(hi), "mean": float(v.mean()), "std": float(v.std())}


def run(cfg: dict, num_workers: int = 4) -> dict:
    d = cfg["data"]
    out_dir = Path(d["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = find_cases(d["raw_dir"], d["format"])
    print(f"Found {len(cases)} labelled cases in {d['raw_dir']}")

    splits_path = out_dir / "splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text())
        print(f"Re-using existing split file {splits_path}")
    else:
        splits = make_splits([c[0] for c in cases], d["num_folds"], d["test_fraction"], d["seed"])
        splits_path.write_text(json.dumps(splits, indent=2))

    jobs = [(cid, img, seg, d, out_dir) for cid, img, seg in cases]
    results = []
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for r in pool.imap_unordered(preprocess_case, jobs):
                if r:
                    results.append(r)
                    print(f"[ok] {r['case_id']}  shape={r['shape']}")
    else:
        for j in jobs:
            r = preprocess_case(j)
            if r:
                results.append(r)
                print(f"[ok] {r['case_id']}  shape={r['shape']}")

    test = set(splits["test"])
    train_vals = [r["fg_values"] for r in results if r["case_id"] not in test]
    stats = intensity_statistics(np.concatenate(train_vals), d["clip_hu"])
    spacings = np.array([r["spacing"] for r in results])
    fingerprint = {
        "intensity": stats,
        "target_spacing": d["target_spacing"],
        "median_source_spacing": np.median(spacings, axis=0).tolist(),
        "num_classes": d["num_classes"],
        "roi_crop": bool(d.get("roi_crop")),
        "roi_margin_mm": d["roi_margin_mm"],
        "cases": sorted((r["case_id"] for r in results), key=_natural_key),
    }
    (out_dir / "fingerprint.json").write_text(json.dumps(fingerprint, indent=2))
    print(f"Intensity stats (training patients only): {stats}")
    print(f"Wrote {len(results)} cases to {out_dir}")
    return fingerprint


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=False)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("overrides", nargs="*", help="config overrides like data.raw_dir=/path")
    args = ap.parse_args(argv)
    run(load_config(args.config, args.overrides), args.num_workers)


if __name__ == "__main__":
    main()
