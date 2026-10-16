"""Held-out evaluation at the ORIGINAL scan resolution.

Mode A - run a trained model on the held-out test patients (from splits.json):
    liverseg-evaluate --config configs/lits_3d_fullres.yaml --model outputs/lits_3d_fullres --out results/test

Mode B - score any existing predictions (e.g. nnU-Net output) against ground truth:
    liverseg-evaluate --pred-dir nnunet_preds/ --gt-dir data/raw/labelsTr --out results/nnunet

Writes per_case.csv, summary.json and summary.md (mean, SD, median, bootstrap 95% CI,
LiTS-style global Dice, lesion detection rates).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .inference import Predictor, find_checkpoints
from .io import case_name, orient, read_image, read_label, resample_label_to, to_array, write_image
from .metrics import bootstrap_ci, evaluate_case, global_dice

KEY_METRICS = [
    ("liver_dice", "Liver Dice", "all"),
    ("liver_iou", "Liver IoU", "all"),
    ("liver_hd95_mm", "Liver HD95 (mm)", "all"),
    ("liver_assd_mm", "Liver ASSD (mm)", "all"),
    ("liver_rvd", "Liver RVD", "all"),
    ("liver_voxel_accuracy", "Liver voxel accuracy*", "all"),
    ("tumor_dice", "Tumour Dice", "tumor"),
    ("tumor_iou", "Tumour IoU", "tumor"),
    ("tumor_hd95_mm", "Tumour HD95 (mm)", "tumor"),
    ("tumor_assd_mm", "Tumour ASSD (mm)", "tumor"),
    ("tumor_lesion_sensitivity", "Lesion detection sensitivity", "tumor"),
    ("tumor_lesion_precision", "Lesion detection precision", "all"),
    ("tumor_small_lesion_sensitivity", "Small (<10 mm) lesion sensitivity", "tumor"),
    ("multiclass_voxel_accuracy", "3-class voxel accuracy*", "all"),
]


def score_pair(pred_img, gt_img, distances=True):
    pred_img = resample_label_to(pred_img, gt_img)  # identical grid unless the producer changed it
    gt_o, pred_o = orient(gt_img), orient(pred_img)
    gt, spacing = to_array(gt_o)
    pred, _ = to_array(pred_o)
    gt = np.minimum(gt, 2)
    return evaluate_case(pred.astype(np.uint8), gt.astype(np.uint8), spacing, distances)


def summarize(rows: list) -> dict:
    df = pd.DataFrame(rows)
    summary = {"n_cases": len(df), "n_cases_with_tumor": int(df["tumor_gt_present"].sum()), "metrics": {}}
    for key, label, subset in KEY_METRICS:
        if key not in df:
            continue
        v = df[key] if subset == "all" else df.loc[df["tumor_gt_present"] > 0, key]
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        if len(v) == 0:
            continue
        lo, hi = bootstrap_ci(v.values)
        summary["metrics"][key] = {
            "label": label, "n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(v.median()), "ci95": [lo, hi],
        }
    summary["liver_dice_global"] = global_dice(rows, "liver")
    summary["tumor_dice_global"] = global_dice(rows, "tumor")
    no_tumor = df[df["tumor_gt_present"] == 0]
    if len(no_tumor):
        summary["tumor_free_cases_without_false_positive"] = float((no_tumor["tumor_pred_present"] == 0).mean())
    summary["lesions_total"] = int(df["tumor_lesions_gt"].sum())
    summary["lesions_detected_total"] = int(df["tumor_lesions_detected"].sum())
    return summary


def to_markdown(s: dict) -> str:
    f = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"
    lines = [
        f"# Evaluation ({s['n_cases']} cases, {s['n_cases_with_tumor']} with tumour)",
        "",
        "| Metric | n | Mean ± SD | Median | 95% CI (bootstrap) |",
        "|---|---|---|---|---|",
    ]
    for m in s["metrics"].values():
        lines.append(f"| {m['label']} | {m['n']} | {f(m['mean'])} ± {f(m['sd'])} | {f(m['median'])} | {f(m['ci95'][0])} - {f(m['ci95'][1])} |")
    lines += [
        "",
        f"- Global Dice (LiTS-style, pooled voxels): liver {f(s['liver_dice_global'])}, tumour {f(s['tumor_dice_global'])}",
        f"- Lesions detected: {s['lesions_detected_total']} / {s['lesions_total']}",
    ]
    if "tumor_free_cases_without_false_positive" in s:
        lines.append(f"- Tumour-free scans with no false-positive tumour: {f(s['tumor_free_cases_without_false_positive'])}")
    lines += [
        "",
        "\\* Voxel accuracy is dominated by background (most of a CT is not liver), so it is",
        "close to 1 even for poor segmentations. Judge the model on Dice, HD95 and lesion detection.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--model", nargs="*")
    ap.add_argument("--locator", nargs="*")
    ap.add_argument("--split", default="test", help="'test' or 'val:<fold>'")
    ap.add_argument("--pred-dir")
    ap.add_argument("--gt-dir")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-distances", action="store_true", help="skip HD95/ASSD (faster)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    if a.pred_dir:
        gts = {case_name(p).replace("segmentation-", ""): p for p in Path(a.gt_dir).glob("*.nii*")}
        for p in sorted(Path(a.pred_dir).glob("*.nii*")):
            key = case_name(p).replace("_seg", "").replace("segmentation-", "").replace("volume-", "")
            if key not in gts:
                print(f"[skip] no ground truth for {p.name}")
                continue
            r = score_pair(read_label(p), read_label(gts[key]), not a.no_distances)
            r["case"] = key
            rows.append(r)
            print(f"{key}: liver {r['liver_dice']:.4f} tumour {r['tumor_dice']:.4f}")
    else:
        cfg = load_config(a.config, a.overrides)
        pp = Path(cfg["data"]["preprocessed_dir"])
        splits = json.loads((pp / "splits.json").read_text())
        ids = splits["test"] if a.split == "test" else splits["folds"][int(a.split.split(":")[1])]["val"]
        ckpts = [c for m in a.model for c in find_checkpoints(m)]
        locator = Predictor([c for m in a.locator for c in find_checkpoints(m)], device=a.device, tta=False) if a.locator else None
        predictor = Predictor(ckpts, device=a.device, tta=not a.no_tta, locator=locator)
        for cid in ids:
            meta_path = pp / f"{cid}.pkl"
            if not meta_path.exists():
                continue
            meta = pickle.loads(meta_path.read_bytes())
            img = read_image(meta["image_path"])
            gt = read_label(meta["label_path"])
            gt.CopyInformation(img)
            pred = predictor.predict(img)
            write_image(pred.label, out / "predictions" / f"{cid}_seg.nii.gz")
            r = score_pair(pred.label, gt, not a.no_distances)
            r["case"] = cid
            rows.append(r)
            print(f"{cid}: liver {r['liver_dice']:.4f} tumour {r['tumor_dice']:.4f} lesions {int(r['tumor_lesions_detected'])}/{int(r['tumor_lesions_gt'])}")

    if not rows:
        raise SystemExit("No cases evaluated")
    pd.DataFrame(rows).set_index("case").to_csv(out / "per_case.csv")
    s = summarize(rows)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    md = to_markdown(s)
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
