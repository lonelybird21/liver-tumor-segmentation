"""Model factory."""

from __future__ import annotations

import torch.nn as nn

from ..config import in_channels
from .unet import UNet


def build_model(cfg: dict) -> nn.Module:
    m = cfg["model"]
    nc = int(cfg["data"]["num_classes"])
    cin = in_channels(cfg)
    name = m["name"].lower()
    if name in ("resunet", "unet"):
        return UNet(
            dims=m["dims"],
            in_channels=cin,
            num_classes=nc,
            base_filters=m["base_filters"],
            max_filters=m["max_filters"],
            n_stages=m["n_stages"],
            blocks_per_stage=m["blocks_per_stage"],
            residual=name == "resunet",
            deep_supervision=m["deep_supervision"],
            attention_gates=m["attention_gates"],
            dropout=m["dropout"],
        )
    try:
        import monai.networks.nets as mn
    except ImportError as e:  # pragma: no cover
        raise ImportError(f"model.name={name!r} needs MONAI: pip install monai") from e
    if name == "segresnet":
        return mn.SegResNet(spatial_dims=m["dims"], in_channels=cin, out_channels=nc, init_filters=m["base_filters"], dropout_prob=m["dropout"] or None)
    if name == "swinunetr":
        kwargs = dict(spatial_dims=m["dims"], in_channels=cin, out_channels=nc, feature_size=48, use_checkpoint=True)
        try:
            return mn.SwinUNETR(img_size=tuple(cfg["train"]["patch_size"][-m["dims"]:]), **kwargs)
        except TypeError:  # MONAI >= 1.5 dropped img_size
            return mn.SwinUNETR(**kwargs)
    raise ValueError(f"Unknown model.name {name!r}")


def uses_deep_supervision(cfg: dict) -> bool:
    return cfg["model"]["name"] in ("resunet", "unet") and bool(cfg["model"]["deep_supervision"])
