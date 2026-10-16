import numpy as np
import pytest
import SimpleITK as sitk
import torch

from liverseg.augment import spatial_transform
from liverseg.config import load_config
from liverseg.inference import probs_to_label, sliding_window_predict, window_starts
from liverseg.io import label_to_original_grid, orient, to_array
from liverseg.losses import build_loss
from liverseg.metrics import binary_metrics, evaluate_case, lesion_detection
from liverseg.models import build_model
from liverseg.postprocess import postprocess
from liverseg.preprocessing import make_splits, resample_array
from liverseg.report import longest_axial_diameter, summarize


def sphere(shape, center, r):
    g = np.indices(shape)
    return sum((g[i] - center[i]) ** 2 for i in range(len(shape))) <= r * r


@pytest.mark.parametrize("dims,patch", [(3, [16, 32, 32]), (2, [1, 64, 64])])
def test_model_shapes_and_deep_supervision(dims, patch):
    cfg = load_config(None, [f"model.dims={dims}", "model.n_stages=4", "model.blocks_per_stage=[1,1,1,1]",
                             "model.base_filters=4", f"train.patch_size={patch}", "model.attention_gates=true"])
    m = build_model(cfg)
    spatial = patch if dims == 3 else patch[1:]
    x = torch.randn(2, 1, *spatial)
    m.train()
    outs = m(x)
    assert isinstance(outs, list) and outs[0].shape == (2, 3, *spatial)
    assert outs[1].shape[2:] == tuple(s // 2 for s in spatial)
    y = torch.randint(0, 3, (2, *spatial))
    loss = build_loss(cfg)(outs, y)
    loss.backward()
    m.eval()
    assert m(x).shape == (2, 3, *spatial)


def test_config_rejects_bad_patch():
    with pytest.raises(ValueError):
        load_config(None, ["train.patch_size=[30,64,64]"])


def test_sliding_window_reconstructs_identity():
    class OneHot(torch.nn.Module):  # "model" whose softmax reproduces a label encoded in the input
        def forward(self, x):
            return torch.cat([-(x * 50), x * 50, torch.zeros_like(x) - 100], 1)

    lab = np.zeros((20, 50, 45), np.float32)
    lab[5:15, 10:40, 8:30] = 1
    probs = sliding_window_predict(OneHot(), lab[None], [8, 16, 16], 3, overlap=0.5, mirror_axes=(0, 2))
    assert probs.shape == (3, 20, 50, 45)
    assert np.array_equal(probs.argmax(0), lab.astype(int))


def test_window_starts_cover_volume():
    for size, patch in [(10, 16), (100, 32), (33, 32)]:
        st = window_starts(size, patch, 0.5)
        assert st[0] == 0 and st[-1] + patch >= size


def test_resample_roundtrip_and_label_mode():
    lab = sphere((30, 60, 60), (15, 30, 30), 10).astype(np.uint8) * 2
    down = resample_array(lab, (1, 1, 1), (2, 2, 2), is_label=True)
    assert down.shape == (15, 30, 30) and set(np.unique(down)) <= {0, 2}
    probs = np.stack([(down == c).astype(np.float32) for c in range(3)])
    up = probs_to_label(probs, lab.shape)
    assert binary_metrics(up > 0, lab > 0, (1, 1, 1), False)["dice"] > 0.93  # 2x down+up of a small sphere


def test_orientation_roundtrip():
    arr = np.random.default_rng(0).integers(0, 3, (7, 9, 11)).astype(np.uint8)
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((0.7, 0.8, 2.5))
    for code in ("RAS", "LPI", "RSA"):
        ref = sitk.DICOMOrient(img, code)
        o = orient(ref)
        lab_o, _ = to_array(sitk.Cast(o, sitk.sitkUInt8))
        back = label_to_original_grid(lab_o, o, ref)
        assert np.array_equal(sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(sitk.Cast(ref, sitk.sitkUInt8)))


def test_metrics_known_values():
    gt = sphere((40, 40, 40), (20, 20, 20), 8)
    m = binary_metrics(gt, gt, (1, 1, 1))
    assert m["dice"] == 1 and m["hd95_mm"] == 0
    shifted = np.roll(gt, 3, axis=2)
    m = binary_metrics(shifted, gt, (1, 1, 1))
    assert 0.5 < m["dice"] < 0.9 and 2.0 <= m["hd95_mm"] <= 3.0
    # voxel accuracy stays ~1 even when the prediction is empty: why we don't use it
    empty = binary_metrics(np.zeros_like(gt), gt, (1, 1, 1))
    assert empty["dice"] == 0 and empty["voxel_accuracy"] > 0.96


def test_lesion_detection_counts():
    gt = sphere((30, 60, 60), (15, 15, 15), 5) | sphere((30, 60, 60), (15, 45, 45), 5)
    pred = sphere((30, 60, 60), (15, 15, 16), 5) | sphere((30, 60, 60), (15, 45, 10), 3)
    r = lesion_detection(pred, gt, (1, 1, 1))
    assert r["lesions_gt"] == 2 and r["lesions_detected"] == 1 and r["false_positive_lesions"] == 1


def test_postprocess_removes_islands_and_outside_tumour():
    lab = np.zeros((30, 60, 60), np.uint8)
    lab[sphere(lab.shape, (15, 30, 30), 12)] = 1
    lab[sphere(lab.shape, (15, 30, 30), 3)] = 2
    lab[2:4, 2:4, 2:4] = 1          # liver false-positive island
    lab[25:27, 55:57, 55:57] = 2    # tumour outside the liver
    out = postprocess(lab, (1, 1, 1), min_tumor_volume_mm3=0)
    assert out[3, 3, 3] == 0 and out[26, 56, 56] == 0 and out[15, 30, 30] == 2


def test_report_measurements():
    lab = np.zeros((20, 100, 100), np.uint8)
    lab[2:18, 10:90, 10:90] = 1
    lab[8:12, 40:60, 40:50] = 2  # 20 x 10 voxels in-plane
    s = summarize(lab, (2.0, 1.0, 1.0))
    assert s["number_of_lesions"] == 1
    assert abs(s["lesions"][0]["longest_axial_diameter_mm"] - np.hypot(20, 10)) < 0.051  # rounded to 0.1 mm
    assert abs(s["tumor_volume_ml"] - 4 * 20 * 10 * 2 / 1000) < 1e-6


def test_evaluate_case_keys():
    lab = np.zeros((10, 30, 30), np.uint8)
    lab[2:8, 5:25, 5:25] = 1
    lab[4:6, 10:14, 10:14] = 2
    r = evaluate_case(lab, lab, (1, 1, 1))
    assert r["liver_dice"] == 1 and r["tumor_dice"] == 1 and r["tumor_lesion_sensitivity"] == 1


def test_splits_have_no_leakage():
    ids = [f"case_{i}" for i in range(131)]
    s = make_splits(ids, 5, 0.15, 42)
    assert len(s["test"]) == 20
    for f in s["folds"]:
        assert not set(f["train"]) & set(f["val"])
        assert not (set(f["train"]) | set(f["val"])) & set(s["test"])
    assert sorted(c for f in s["folds"] for c in f["val"]) == sorted(set(ids) - set(s["test"]))


def test_spatial_transform_identity():
    img = torch.randn(1, 12, 20, 20)
    lab = (img[0] > 0).long()
    out, ol = spatial_transform(img, lab, (12, 20, 20))
    assert torch.allclose(out, img, atol=1e-4) and torch.equal(ol, lab)
