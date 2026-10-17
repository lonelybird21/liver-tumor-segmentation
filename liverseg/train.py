"""Training with patch sampling, deep supervision, mixed precision, poly/cosine LR,
full-volume sliding-window validation, checkpoint/resume and early stopping.

Usage:
    python -m liverseg.train --config configs/lits_3d_fullres.yaml --fold 0
    python -m liverseg.train --config configs/lits_3d_fullres.yaml --fold 0 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import CaseStore, PatchDataset, full_case, load_fingerprint, load_splits
from .inference import sliding_window_predict
from .losses import build_loss
from .metrics import binary_metrics
from .models import build_model


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(epoch: int, cfg: dict) -> float:
    t = cfg["train"]
    base, total, warm = t["lr"], t["epochs"], t["warmup_epochs"]
    if warm and epoch < warm:
        return base * (epoch + 1) / warm
    progress = (epoch - warm) / max(1, total - warm)
    if t["scheduler"] == "cosine":
        return base * 0.5 * (1 + math.cos(math.pi * progress))
    return base * (1 - progress) ** 0.9


def build_optimizer(model, cfg):
    t = cfg["train"]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"], nesterov=True, weight_decay=t["weight_decay"])


@torch.no_grad()
def validate(model, store, case_ids, cfg, intensity, device):
    """Full-volume sliding-window prediction on each validation case (at model spacing)."""
    model.eval()
    dims = cfg["model"]["dims"]
    nc = int(cfg["data"]["num_classes"])
    k = int(cfg["model"]["context_slices"])
    rows = []
    for cid in case_ids:
        x, gt = full_case(store, cid, intensity)
        if dims == 3:
            probs = sliding_window_predict(model, x[None], cfg["train"]["patch_size"], nc, 0.5, (), cfg["inference"]["batch_size"], device)
        else:
            probs = np.zeros((nc, *x.shape), np.float32)
            for z in range(x.shape[0]):
                idx = np.clip(np.arange(z - k, z + k + 1), 0, x.shape[0] - 1)
                probs[:, z] = sliding_window_predict(model, x[idx], cfg["train"]["patch_size"][1:], nc, 0.5, (), 8, device)
        pred = probs.argmax(0)
        sp = store.meta[cid]["spacing"]
        liver = binary_metrics(pred > 0, gt > 0, sp, distances=False)
        tumor = binary_metrics(pred == 2, gt == 2, sp, distances=False)
        rows.append({"case": cid, "liver_dice": liver["dice"], "tumor_dice": tumor["dice"], "tumor_present": tumor["gt_present"]})
    liver = float(np.mean([r["liver_dice"] for r in rows]))
    with_tumor = [r["tumor_dice"] for r in rows if r["tumor_present"]]
    tumor = float(np.mean(with_tumor)) if with_tumor else float("nan")
    return {"liver_dice": liver, "tumor_dice": tumor, "score": float(np.nanmean([liver, tumor])), "cases": rows}


def save_checkpoint(path: Path, **state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(cfg: dict, fold, resume: bool = False, device: str | None = None) -> Path:
    t = cfg["train"]
    seed_everything(t["seed"] + (0 if fold == "all" else int(fold)))
    pp = Path(cfg["data"]["preprocessed_dir"])
    fp = load_fingerprint(pp)
    splits = load_splits(pp)
    available = set(fp["cases"])
    if fold == "all":
        train_ids = sorted(set(splits["folds"][0]["train"]) | set(splits["folds"][0]["val"]))
        val_ids = splits["folds"][0]["val"]  # monitoring only (these are also trained on)
    else:
        train_ids, val_ids = splits["folds"][int(fold)]["train"], splits["folds"][int(fold)]["val"]
    train_ids = [c for c in train_ids if c in available]
    val_ids = [c for c in val_ids if c in available]
    if t["val_max_cases"]:
        val_ids = val_ids[: int(t["val_max_cases"])]
    if set(train_ids) & set(splits["test"]):
        raise RuntimeError("Test patients leaked into training set")

    out_dir = Path(t["output_root"]) / cfg["experiment"] / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Fold {fold}: {len(train_ids)} train / {len(val_ids)} val cases, device={dev}")

    intensity = fp["intensity"]
    train_store, val_store = CaseStore(pp, train_ids), CaseStore(pp, val_ids)
    ds = PatchDataset(train_store, cfg, intensity, length=t["iters_per_epoch"] * t["batch_size"], augment=True, seed=t["seed"])
    loader = DataLoader(
        ds, batch_size=t["batch_size"], shuffle=False, num_workers=t["num_workers"],
        pin_memory=dev.type == "cuda", persistent_workers=t["num_workers"] > 0, drop_last=True,
    )

    model = build_model(cfg).to(dev)
    loss_fn = build_loss(cfg).to(dev)
    opt = build_optimizer(model, cfg)
    use_amp = bool(t["amp"]) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best, ema, bad_rounds = 0, -1.0, None, 0
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    latest = out_dir / "checkpoint_latest.pt"
    if resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, ema, bad_rounds = ck["epoch"] + 1, ck["best_score"], ck["ema_score"], ck["bad_rounds"]
        print(f"Resumed from epoch {ck['epoch']} (best score {best:.4f})")

    log_path = out_dir / "training_log.csv"
    new_log = not log_path.exists() or not resume
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["epoch", "lr", "train_loss", "val_liver_dice", "val_tumor_dice", "val_score", "ema_score", "seconds"])

    def state(epoch):
        return dict(
            model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
            best_score=best, ema_score=ema, bad_rounds=bad_rounds, config=cfg, intensity=intensity, fold=fold,
        )

    for epoch in range(start_epoch, t["epochs"]):
        t0 = time.time()
        lr = lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        v = {"liver_dice": float("nan"), "tumor_dice": float("nan"), "score": float("nan")}
        is_last = epoch == t["epochs"] - 1
        if val_ids and ((epoch + 1) % t["val_every"] == 0 or is_last):
            v = validate(model, val_store, val_ids, cfg, intensity, dev)
            ema = v["score"] if ema is None else 0.9 * ema + 0.1 * v["score"]
            # pick the checkpoint on the raw validation score; EMA is logged for trend reading
            if v["score"] > best:
                best, bad_rounds = v["score"], 0
                save_checkpoint(out_dir / "checkpoint_best.pt", **state(epoch))
            else:
                bad_rounds += 1
        secs = time.time() - t0
        print(
            f"epoch {epoch:4d} lr {lr:.5f} loss {train_loss:.4f} | val liver {v['liver_dice']:.4f} "
            f"tumor {v['tumor_dice']:.4f} score {v['score']:.4f} (best {best:.4f}) | {secs:.0f}s",
            flush=True,
        )
        log.writerow([epoch, lr, train_loss, v["liver_dice"], v["tumor_dice"], v["score"], ema, round(secs, 1)])
        log_f.flush()
        save_checkpoint(latest, **state(epoch))
        if bad_rounds >= t["early_stop_patience"]:
            print(f"Early stopping: no validation improvement in {bad_rounds} rounds")
            break

    log_f.close()
    final = out_dir / "checkpoint_final.pt"
    save_checkpoint(final, **state(epoch))
    if not (out_dir / "checkpoint_best.pt").exists():
        save_checkpoint(out_dir / "checkpoint_best.pt", **state(epoch))
    print(f"Done. Best validation score {best:.4f}. Checkpoints in {out_dir}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--fold", default="0", help="fold index or 'all'")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    train(load_config(a.config, a.overrides), a.fold, a.resume, a.device)


if __name__ == "__main__":
    main()
