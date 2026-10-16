"""Web demo: upload a CT (NIfTI, zipped DICOM folder, or a PNG/JPG slice for 2D models)
and get the segmentation overlay, volumes, lesion table and a downloadable mask.

    pip install gradio
    python app.py --model outputs/lits_3d_fullres [--locator outputs/lits_3d_lowres]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from liverseg.inference import Predictor, find_checkpoints
from liverseg.io import case_name, is_picture, read_image, write_image
from liverseg.report import DISCLAIMER, render_overlays, summarize


def build(predictor: Predictor):
    import gradio as gr

    def run(file, level, width, spacing):
        if file is None:
            raise gr.Error("Upload a scan first")
        path = Path(file if isinstance(file, str) else file.name)
        work = Path(tempfile.mkdtemp())
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                z.extractall(work / "dicom")
            src = work / "dicom"
        else:
            src = path
        kw = dict(window_level=level, window_width=width, pixel_spacing=spacing) if is_picture(src) else {}
        try:
            pred = predictor.predict(read_image(src, **kw))
        except ValueError as e:
            raise gr.Error(str(e))
        name = case_name(path)
        overlay = work / f"{name}_overlay.png"
        render_overlays(pred.image_oriented, pred.label_oriented, pred.spacing, overlay, level, width)
        mask = work / f"{name}_seg.nii.gz"
        write_image(pred.label, mask)
        s = summarize(pred.label_oriented, pred.spacing)
        table = pd.DataFrame(s.pop("lesions")) if s["number_of_lesions"] else pd.DataFrame()
        return str(overlay), json.dumps(s, indent=2), table, str(mask)

    with gr.Blocks(title="Liver & tumour segmentation") as demo:
        gr.Markdown(f"## Liver & liver-tumour CT segmentation\n{DISCLAIMER}")
        with gr.Row():
            with gr.Column():
                f = gr.File(label="CT: .nii/.nii.gz, .zip of a DICOM series, or PNG/JPG slice (2D model)")
                level = gr.Number(50, label="Picture only: window level (HU)")
                width = gr.Number(400, label="Picture only: window width (HU)")
                spacing = gr.Number(0.8, label="Picture only: pixel spacing (mm)")
                btn = gr.Button("Segment", variant="primary")
            with gr.Column():
                img = gr.Image(label="Overlay")
                summary = gr.Code(label="Summary", language="json")
        lesions = gr.Dataframe(label="Lesions")
        out = gr.File(label="Mask (NIfTI, original grid)")
        btn.click(run, [f, level, width, spacing], [img, summary, lesions, out])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--locator", nargs="*")
    ap.add_argument("--device")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    loc = Predictor([c for m in a.locator for c in find_checkpoints(m)], device=a.device, tta=False) if a.locator else None
    p = Predictor([c for m in a.model for c in find_checkpoints(m)], device=a.device, tta=not a.no_tta, locator=loc)
    build(p).launch(share=a.share)
