"""Whole pipeline on tiny synthetic phantoms: preprocess -> train -> predict -> evaluate."""

import json

import numpy as np
import SimpleITK as sitk

from liverseg import evaluate, predict, preprocessing, synthetic, train
from liverseg.config import load_config


def test_pipeline_runs(tmp_path):
    raw = synthetic.write_dataset(tmp_path / "raw", n=6, shape=(24, 64, 64))
    over = [
        f"data.raw_dir={raw}", f"data.preprocessed_dir={tmp_path / 'pp'}", f"train.output_root={tmp_path / 'out'}",
        "experiment=e2e", "data.num_folds=2", "data.test_fraction=0.34", "data.target_spacing=[3.0,2.5,2.5]",
        "model.base_filters=4", "model.max_filters=16", "model.n_stages=3", "model.blocks_per_stage=[1,1,1]",
        "train.patch_size=[8,32,32]", "train.epochs=2", "train.iters_per_epoch=2", "train.num_workers=0",
        "train.val_every=1", "train.amp=false", "inference.tta=true",
    ]
    cfg = load_config(None, over)
    preprocessing.run(cfg, num_workers=1)
    out_dir = train.train(cfg, "0", device="cpu")
    assert (out_dir / "checkpoint_best.pt").exists()

    # resume continues from the saved epoch without error
    cfg2 = load_config(None, over + ["train.epochs=3"])
    train.train(cfg2, "0", resume=True, device="cpu")

    pred_dir = tmp_path / "pred"
    predict.main([str(raw / "volume-1.nii.gz"), "--model", str(out_dir), "--out", str(pred_dir), "--report", "--device", "cpu"])
    seg = sitk.ReadImage(str(pred_dir / "volume-1_seg.nii.gz"))
    ref = sitk.ReadImage(str(raw / "volume-1.nii.gz"))
    assert seg.GetSize() == ref.GetSize() and seg.GetDirection() == ref.GetDirection()
    assert (pred_dir / "volume-1_overlay.png").exists()

    res = tmp_path / "res"
    evaluate.main(["--model", str(out_dir), "--out", str(res), "--device", "cpu", *over])
    s = json.loads((res / "summary.json").read_text())
    assert s["n_cases"] >= 1 and "liver_dice" in s["metrics"]
    assert np.isfinite(s["metrics"]["liver_dice"]["mean"])
