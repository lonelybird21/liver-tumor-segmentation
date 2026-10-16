"""Training data: memory-mapped preprocessed cases, class-aware patch sampling, augmentation."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from . import augment as A
from .preprocessing import normalize_hu

PAD_HU = -1024


class CaseStore:
    """Lazily opens preprocessed cases as read-only memory maps (cheap in each worker)."""

    def __init__(self, preprocessed_dir: str | Path, case_ids: Sequence[str]):
        self.root = Path(preprocessed_dir)
        self.case_ids = list(case_ids)
        self.meta: Dict[str, dict] = {}
        for cid in self.case_ids:
            with open(self.root / f"{cid}.pkl", "rb") as f:
                self.meta[cid] = pickle.load(f)
        self._arrays: Dict[str, tuple] = {}

    def arrays(self, cid: str):
        if cid not in self._arrays:
            img = np.load(self.root / f"{cid}_img.npy", mmap_mode="r")
            seg = np.load(self.root / f"{cid}_seg.npy", mmap_mode="r")
            self._arrays[cid] = (img, seg)
        return self._arrays[cid]

    def __getstate__(self):  # do not pickle open memmaps into workers
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state


def load_fingerprint(preprocessed_dir) -> dict:
    return json.loads((Path(preprocessed_dir) / "fingerprint.json").read_text())


def load_splits(preprocessed_dir) -> dict:
    return json.loads((Path(preprocessed_dir) / "splits.json").read_text())


def crop_with_pad(arr: np.ndarray, center: Sequence[int], size: Sequence[int], pad_value) -> np.ndarray:
    out = np.full(tuple(size), pad_value, dtype=arr.dtype)
    src, dst = [], []
    for c, s, n in zip(center, size, arr.shape):
        lo = int(c) - s // 2
        hi = lo + s
        s_lo, s_hi = max(lo, 0), min(hi, n)
        src.append(slice(s_lo, s_hi))
        dst.append(slice(s_lo - lo, s_lo - lo + max(0, s_hi - s_lo)))
    out[tuple(dst)] = arr[tuple(src)]
    return out


class PatchDataset(Dataset):
    """Infinite-style random patch sampler.

    Each item picks a random case and a patch centre. With probability
    `fg_oversample` the centre lies on foreground; of those, `tumor_oversample`
    are centred on tumour when the case has any. Without this, random 3D patches
    almost never contain tumour and the network learns to ignore it.
    """

    def __init__(self, store: CaseStore, cfg: dict, intensity: dict, length: int, augment: bool = True, seed: int = 0):
        self.store = store
        self.cfg = cfg
        self.intensity = intensity
        self.length = length
        self.augment = augment and cfg["train"]["augmentation"]["enabled"]
        self.seed = seed
        self.dims = cfg["model"]["dims"]
        k = int(cfg["model"]["context_slices"])
        p = list(cfg["train"]["patch_size"])
        # what we crop from the volume (z, y, x); for 2D the z extent is the context stack
        self.patch_zyx = [2 * k + 1, p[1], p[2]] if self.dims == 2 else p
        self.context = k

    def __len__(self):
        return self.length

    def _choose_center(self, rng, meta, shape):
        t = self.cfg["train"]
        locs = meta["class_locations"]
        if rng.random() < t["fg_oversample"]:
            tumor = locs.get(2)
            if tumor is not None and len(tumor) and rng.random() < t["tumor_oversample"]:
                pts = tumor
            else:
                pts = np.concatenate([v for v in locs.values() if len(v)] or [np.zeros((0, 3), int)])
            if len(pts):
                return pts[rng.integers(len(pts))]
        # uniform centre, kept far enough inside that most of the patch is real data
        out = []
        for s, n in zip(self.patch_zyx, shape):
            lo, hi = min(s // 2, n // 2), max(n - s // 2, n // 2 + 1)
            out.append(rng.integers(lo, hi))
        return np.array(out)

    def __getitem__(self, idx):
        rng = np.random.default_rng([self.seed, idx, torch.initial_seed() % 2**31])
        cid = self.store.case_ids[rng.integers(len(self.store.case_ids))]
        meta = self.store.meta[cid]
        img_arr, seg_arr = self.store.arrays(cid)
        center = self._choose_center(rng, meta, img_arr.shape)

        a = self.cfg["train"]["augmentation"]
        do_rot = self.augment and rng.random() < a["p_rotation"]
        do_scale = self.augment and rng.random() < a["p_scale"]
        angles = [math.radians(rng.uniform(-d, d)) if do_rot else 0.0 for d in a["rotation_deg"]]
        scale = rng.uniform(*a["scale_range"]) if do_scale else 1.0
        spatial = do_rot or do_scale

        crop = list(self.patch_zyx)
        if spatial:
            grow = 1.5 / min(scale, 1.0)
            if self.dims == 3:
                crop = [int(math.ceil(s * grow)) for s in crop]
            else:
                crop = [crop[0]] + [int(math.ceil(s * grow)) for s in crop[1:]]

        img = crop_with_pad(img_arr, center, crop, PAD_HU)
        seg = crop_with_pad(seg_arr, center, crop, 0)
        img = torch.from_numpy(normalize_hu(img, self.intensity))
        seg = torch.from_numpy(seg.astype(np.int64))

        if self.dims == 3:
            img = img[None]                       # (1, z, y, x)
        else:
            seg = seg[self.context]                # centre slice label (y, x); img is (2k+1, y, x)

        if spatial:
            out = self.patch_zyx if self.dims == 3 else self.patch_zyx[1:]
            img, seg = A.spatial_transform(img, seg, out, angles, scale)

        if self.augment:
            axes = [ax for ax in a["mirror_axes"] if self.dims == 3 or ax > 0]
            if self.dims == 2:
                axes = [ax - 1 for ax in axes]
            img, seg = A.mirror(img, seg, rng, axes, a["p_mirror"])
            img = A.intensity_augment(img, rng, a)

        return img.contiguous().float(), seg.contiguous().long()


def full_case(store: CaseStore, cid: str, intensity: dict):
    """Whole preprocessed case as a normalized float array and its label (for validation)."""
    img, seg = store.arrays(cid)
    return normalize_hu(np.asarray(img), intensity), np.asarray(seg)
