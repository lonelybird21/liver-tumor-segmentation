"""Segment new CT scans.

Inputs: NIfTI/NRRD/MHA files, DICOM series folders, single DICOM files, or
(2D model only) PNG/JPG pictures of an axial CT slice.

Examples:
    # 5-fold ensemble with mirror TTA, writes masks + clinical reports
    liverseg-predict scan.nii.gz --model outputs/lits_3d_fullres --out predictions --report

    # two-stage cascade: coarse locator finds the liver, fine model segments it
    liverseg-predict dicom_folder/ --locator outputs/lits_3d_lowres --model outputs/lits_3d_roi --out predictions

    # a single picture of a CT slice (needs a 2D model)
    liverseg-predict slice.png --model outputs/lits_2d --window-level 50 --window-width 400 --pixel-spacing 0.75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .inference import Predictor
from .io import case_name, is_picture, read_image, write_image
from .report import render_overlays, summarize, write_report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--model", required=True, nargs="+", help="checkpoint file(s) or experiment dir (all folds are ensembled)")
    ap.add_argument("--locator", nargs="*", help="coarse model for the cascade (optional)")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--device")
    ap.add_argument("--no-tta", action="store_true", help="disable mirror test-time augmentation (faster)")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--report", action="store_true", help="write volumes / lesion table / overlay figure")
    ap.add_argument("--window-level", type=float, default=50.0, help="pictures only: display window used to render the image")
    ap.add_argument("--window-width", type=float, default=400.0)
    ap.add_argument("--pixel-spacing", type=float, default=0.8, help="pictures only: mm per pixel")
    a = ap.parse_args(argv)

    from .inference import find_checkpoints

    ckpts = [c for m in a.model for c in find_checkpoints(m)]
    locator = None
    if a.locator:
        locator = Predictor([c for m in a.locator for c in find_checkpoints(m)], device=a.device, tta=False)
    predictor = Predictor(ckpts, device=a.device, tta=not a.no_tta, locator=locator, do_postprocess=not a.no_postprocess)
    print(f"Loaded {len(ckpts)} model(s) ({predictor.dims}D) on {predictor.device}; TTA={predictor.tta}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for inp in a.inputs:
        name = case_name(inp)
        img = read_image(inp, window_level=a.window_level, window_width=a.window_width, pixel_spacing=a.pixel_spacing) \
            if is_picture(inp) else read_image(inp)
        pred = predictor.predict(img)
        if is_picture(inp):
            Image.fromarray((pred.label_oriented[0] * 127).astype(np.uint8)).save(out / f"{name}_mask.png")
            render_overlays(pred.image_oriented, pred.label_oriented, pred.spacing, out / f"{name}_overlay.png",
                            a.window_level, a.window_width)
            s = summarize(pred.label_oriented, pred.spacing)
            s["note"] = "Picture input: areas assume the given pixel spacing; volumes are per 1 mm slab."
            (out / f"{name}_report.json").write_text(json.dumps(s, indent=2))
        else:
            write_image(pred.label, out / f"{name}_seg.nii.gz")
            if a.report:
                s = write_report(pred.image_oriented, pred.label_oriented, pred.spacing, out, name)
            else:
                s = summarize(pred.label_oriented, pred.spacing)
        print(f"{name}: liver {s['liver_volume_ml']} mL, tumour {s['tumor_volume_ml']} mL, {s['number_of_lesions']} lesion(s)")


if __name__ == "__main__":
    main()
