"""Configuration: YAML files merged over defaults, with `a.b.c=value` command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

DEFAULTS: dict = {
    "experiment": "lits_3d_fullres",
    "data": {
        "raw_dir": "data/raw",
        "preprocessed_dir": "data/preprocessed/3d_fullres",
        # lits:   volume-<id>.nii(.gz) + segmentation-<id>.nii(.gz)
        # msd:    imagesTr/<case>.nii.gz + labelsTr/<case>.nii.gz   (MSD Task03_Liver)
        # nnunet: imagesTr/<case>_0000.nii.gz + labelsTr/<case>.nii.gz
        "format": "lits",
        # Voxel spacing in (z, y, x) mm. `null` keeps the source spacing for that axis
        # (used by the 2D model, which only resamples in-plane).
        "target_spacing": [1.5, 0.8, 0.8],
        # HU clipping: "auto" = 0.5/99.5 percentiles of liver+tumor voxels in the
        # training cases (nnU-Net style), or an explicit [low, high] window.
        "clip_hu": "auto",
        "num_classes": 3,
        "num_folds": 5,
        "test_fraction": 0.15,
        "seed": 42,
        # Cascade stage 2: crop every case to the liver bounding box (+margin) so the
        # network only learns the liver region. Requires a locator model at inference.
        "roi_crop": False,
        "roi_margin_mm": [15.0, 20.0, 20.0],
    },
    "model": {
        "name": "resunet",           # resunet | unet | segresnet (monai) | swinunetr (monai)
        "dims": 3,                   # 3 = volumetric model, 2 = slice model (accepts single images)
        "context_slices": 0,         # 2D only: k neighbouring slices each side as extra channels (2.5D)
        "base_filters": 32,
        "max_filters": 320,
        "n_stages": 5,
        "blocks_per_stage": [1, 2, 2, 2, 2],
        "deep_supervision": True,
        "attention_gates": False,
        "dropout": 0.0,
    },
    "train": {
        "patch_size": [80, 160, 160],
        "batch_size": 2,
        "epochs": 500,
        "iters_per_epoch": 250,
        "optimizer": "sgd",          # sgd (nesterov) | adamw
        "lr": 0.01,
        "weight_decay": 3.0e-5,
        "momentum": 0.99,
        "scheduler": "poly",         # poly | cosine
        "warmup_epochs": 0,
        "grad_clip": 12.0,
        "amp": True,
        "num_workers": 4,
        # Fraction of patches forced to be centred on foreground, and of those the
        # fraction centred on tumour (the rare class).
        "fg_oversample": 0.4,
        "tumor_oversample": 0.6,
        "val_every": 10,
        "val_max_cases": None,
        "early_stop_patience": 15,   # in validation rounds
        "loss": {
            "dice_weight": 1.0,
            "ce_weight": 1.0,
            "class_weights": None,   # e.g. [1.0, 1.0, 3.0]
            "focal_gamma": 0.0,      # >0 turns CE into focal loss
            "dice_include_background": False,
        },
        "augmentation": {
            "enabled": True,
            "p_rotation": 0.2,
            "rotation_deg": [0.0, 0.0, 15.0],   # around (x, y, z) axes; z = in-plane
            "p_scale": 0.2,
            "scale_range": [0.85, 1.25],
            "mirror_axes": [0, 1, 2],           # spatial axes in (z, y, x)
            "p_mirror": 0.5,
            "p_noise": 0.15,
            "noise_std": [0.0, 0.1],
            "p_blur": 0.2,
            "blur_sigma": [0.5, 1.0],
            "p_brightness": 0.15,
            "brightness_range": [0.75, 1.25],
            "p_contrast": 0.15,
            "contrast_range": [0.75, 1.25],
            "p_lowres": 0.25,
            "lowres_range": [0.5, 1.0],
            "p_gamma": 0.3,
            "gamma_range": [0.7, 1.5],
        },
        "output_root": "outputs",
        "seed": 12345,
    },
    "inference": {
        "overlap": 0.5,
        "tta": True,
        "batch_size": 2,
        "postprocess": {
            "liver_largest_cc": True,
            "tumor_inside_liver": True,
            "fill_liver_holes": True,
            "min_tumor_volume_mm3": 10.0,
        },
    },
}


def deep_update(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: Optional[Iterable[str]]) -> dict:
    """Apply `section.key=value` overrides; the value is parsed as YAML."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like a.b=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def validate(cfg: dict) -> dict:
    m, t, d = cfg["model"], cfg["train"], cfg["data"]
    if m["dims"] not in (2, 3):
        raise ValueError("model.dims must be 2 or 3")
    if len(t["patch_size"]) != 3:
        raise ValueError("train.patch_size must have 3 entries (z, y, x); use z=1 for 2D models")
    if m["dims"] == 2 and t["patch_size"][0] != 1:
        raise ValueError("2D models need train.patch_size[0] == 1")
    if len(m["blocks_per_stage"]) != m["n_stages"]:
        raise ValueError("model.blocks_per_stage needs one entry per stage")
    if m["name"] in ("resunet", "unet"):
        factor = 2 ** (m["n_stages"] - 1)
        spatial = t["patch_size"][1:] if m["dims"] == 2 else t["patch_size"]
        for s in spatial:
            if s % factor:
                raise ValueError(
                    f"patch size {t['patch_size']} must be divisible by {factor} "
                    f"(2**(n_stages-1)) on every spatial axis"
                )
    if len(d["target_spacing"]) != 3:
        raise ValueError("data.target_spacing must be (z, y, x)")
    return cfg


def load_config(path: Optional[str | Path] = None, overrides: Optional[Iterable[str]] = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    cfg = apply_overrides(cfg, overrides)
    return validate(cfg)


def in_channels(cfg: dict) -> int:
    m = cfg["model"]
    return 1 if m["dims"] == 3 else 2 * int(m["context_slices"]) + 1
