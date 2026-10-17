# Liver & Liver-Tumour Segmentation from CT

End-to-end pipeline for automatic **liver and liver-tumour segmentation** in contrast-enhanced abdominal CT. It covers preprocessing, training, held-out evaluation and quantitative reporting. Trained on LiTS, and it runs on new scans as NIfTI, DICOM series or (2D model) single slice images.

> **Research software, not a medical device.** Outputs must be reviewed by a qualified radiologist before any clinical use.

## What's new in v2

| Area | v1 | v2 |
|---|---|---|
| Orientation / axes | nibabel (x,y,z) fed as (D,H,W), so patches were mis-shaped | Every scan reoriented to LPS and (z,y,x) used consistently; predictions written back on the **exact original grid** |
| Patch sampling | uniform random, so tumour was almost never seen | **foreground + tumour oversampling** (by default 40% of patches centred on foreground, 60% of those on tumour) |
| Augmentation | none | rotation, scaling, mirroring, noise, blur, brightness, contrast, gamma, low-resolution simulation |
| Model | plain 3D U-Net, 16 filters | **residual U-Net** (2D or 3D) with **deep supervision**, optional **attention gates**, optional MONAI SegResNet / SwinUNETR |
| Training | Adam, model selected on a random patch | SGD-Nesterov + poly LR, mixed precision, gradient clipping, **full-volume sliding-window validation**, resume, early stopping, CSV log |
| Splits | single 70/15/15 | patient-level held-out test set + **5-fold cross-validation**; intensity statistics computed from training patients only |
| Inference | none | Gaussian sliding window, **mirror TTA**, **5-fold ensembling**, **2-stage cascade** (locator then high-res liver ROI), anatomical post-processing |
| Inputs | preprocessed .npy | NIfTI / NRRD / MHA, **DICOM folders**, single DICOM, **PNG/JPG slice** (2D model) |
| Evaluation | patch Dice | Dice, IoU, **HD95, ASSD**, RVD, volumes, **per-lesion detection** (incl. lesions < 10 mm), LiTS global Dice, bootstrap 95% CI, at original resolution |
| Output | none | NIfTI mask, liver volume, tumour burden, **per-lesion table** (volume, RECIST-style long axis), overlay figures, JSON/Markdown report, **web demo** |
| Baseline | none | scripts to train **nnU-Net v2** on the same split and score it with the same evaluator |
| Quality | none | unit + end-to-end tests on synthetic phantoms, GitHub Actions CI |

## About "accuracy"

**Voxel accuracy is not a meaningful number for this task, and 99.99% is not an achievable or honest target.** Background is roughly 90–95% of a CT volume. A model that predicts only background already scores about 95% accuracy, and a mediocre liver model scores above 99%. The test suite asserts this: an *empty* prediction has Dice 0 and voxel accuracy above 96%.

What matters is overlap and boundary error per structure. On LiTS, the best submissions reached about **0.96 Dice for liver** and about **0.74 Dice for tumour** ([Bilic et al., *Med Image Anal* 2023](https://arxiv.org/abs/1901.04056)). Small-lesion detection remains the main open problem. Expert annotators also disagree on tumour borders, so no model can exceed the reliability of its labels. This repository reports voxel accuracy (marked with an asterisk), but you should judge models on **Dice, HD95 and lesion-level sensitivity / false positives**, reported per class.

| Class | Dice (val) | HD95 (mm) | Lesion sensitivity |
|---|---|---|---|
| Liver | *fill in after training* | | |
| Tumour | *fill in after training* | | |

## Install

```bash
git clone https://github.com/lonelybird21/liver-tumor-segmentation.git
cd liver-tumor-segmentation
pip install -e ".[dev]"          # add ,monai for SegResNet/SwinUNETR, ,app for the web demo
pytest -q                        # ~10 s, synthetic data, CPU
```

## Data

Primary: **LiTS**: 131 labelled contrast-enhanced CT volumes from 7 centres (NIfTI, labels 0 = background, 1 = liver, 2 = tumour).

```bash
kaggle datasets download -d andrewmvd/liver-tumor-segmentation
unzip liver-tumor-segmentation.zip -d data/raw/      # volume-N.nii / segmentation-N.nii
```

Also supported via `data.format`: `msd` (Medical Segmentation Decathlon Task03_Liver) and `nnunet` layouts.

For external validation, use **3D-IRCADb-01** (20 patients, different centres and scanners). Convert its DICOM to NIfTI, or point `liverseg-predict` straight at the DICOM folders, then score with `liverseg-evaluate --pred-dir/--gt-dir`. **MSD Task08** (hepatic vessels) and **TCGA-LIHC** (imaging + clinical + pathology) are natural extensions.

## Usage

### 1. Preprocess

```bash
python -m liverseg.preprocessing --config configs/lits_3d_fullres.yaml --num-workers 8
```

This creates `data/preprocessed/3d_fullres/` with memory-mappable volumes, `splits.json` (held-out test set plus 5 folds, split by patient) and `fingerprint.json` (intensity statistics).

### 2. Train

```bash
python -m liverseg.train --config configs/lits_3d_fullres.yaml --fold 0        # repeat for folds 0-4
python -m liverseg.train --config configs/lits_3d_fullres.yaml --fold 0 --resume
# any setting can be overridden on the command line:
python -m liverseg.train --config configs/lits_3d_fullres.yaml --fold 0 train.batch_size=4 model.attention_gates=true
```

Or run everything with `bash scripts/train_all_folds.sh configs/lits_3d_fullres.yaml`.

| Config | Purpose |
|---|---|
| `lits_3d_fullres.yaml` | main 3D model, 1.5 × 0.8 × 0.8 mm, patch 80 × 160 × 160 (roughly 12 GB GPU; shrink patch/filters for smaller cards) |
| `lits_3d_lowres.yaml` | cascade stage 1: fast liver locator at 3 × 2 × 2 mm |
| `lits_3d_roi.yaml` | cascade stage 2: 1.0 × 0.7 × 0.7 mm, liver ROI only, attention gates (best for small tumours) |
| `lits_2d.yaml` | 2D / 2.5D slice model; needed for single images and pictures |
| `smoke_synthetic.yaml` | CPU-sized config for synthetic phantoms (pipeline check only) |

### 3. Evaluate on the held-out test patients

```bash
liverseg-evaluate --config configs/lits_3d_fullres.yaml --model outputs/lits_3d_fullres --out results/fullres
# cascade:
liverseg-evaluate --config configs/lits_3d_roi.yaml --locator outputs/lits_3d_lowres --model outputs/lits_3d_roi --out results/cascade
```

Writes `per_case.csv`, `summary.json` and `summary.md`, all at the original scan resolution.

### 4. Segment a new scan

```bash
liverseg-predict patient.nii.gz          --model outputs/lits_3d_fullres --out predictions --report
liverseg-predict path/to/DICOM_series/   --model outputs/lits_3d_fullres --out predictions --report
liverseg-predict slice.png --model outputs/lits_2d --window-level 50 --window-width 400 --pixel-spacing 0.75
```

`--model` given a directory ensembles every fold it finds. Mirror TTA is on by default (`--no-tta` is about 8× faster).

The `--report` flag writes:

- `*_seg.nii.gz`: the mask on the original grid
- `*_report.md` and `*_report.json`: liver volume, parenchyma volume, tumour volume and burden %, and a lesion table (volume, long-axis diameter, equivalent diameter, slice range)
- `*_overlay.png`: axial and coronal overlays

**About pictures (PNG/JPG):** a screenshot contains display grey levels, not Hounsfield units, and no pixel spacing. The tool reconstructs HU from the display window you specify and assumes the spacing you give, so results are approximate. Use the original DICOM whenever possible.

### 5. Web demo

```bash
pip install gradio
python app.py --model outputs/lits_3d_fullres
```

## nnU-Net baseline

nnU-Net is the reference method for medical segmentation benchmarks. `scripts/run_nnunet_baseline.sh` converts LiTS to nnU-Net v2 format, **keeps the same held-out test patients out of training**, trains 5 folds, and scores the predictions with `liverseg-evaluate`. Both models are then compared on identical metrics and patients.

## Repository structure

```
liverseg/
  config.py          defaults + YAML + CLI overrides
  io.py              NIfTI/DICOM/picture reading, LPS reorientation, write back to original grid
  preprocessing.py   resampling, cropping, splits, intensity fingerprint
  data.py            memory-mapped cases, class-aware patch sampler
  augment.py         spatial + intensity augmentation (2D/3D)
  models/            residual U-Net (deep supervision, attention gates), MONAI wrappers
  losses.py          Dice + (focal) CE, deep-supervision weighting
  train.py           training loop
  inference.py       sliding window, TTA, ensembling, cascade
  postprocess.py     largest component, tumours inside liver, hole filling, small-lesion filter
  metrics.py         Dice/IoU/HD95/ASSD/RVD/lesion detection
  evaluate.py        held-out evaluation with CIs
  report.py          volumes, lesion table, overlays
  predict.py         CLI for new scans
  synthetic.py       synthetic CT phantoms for tests
configs/  scripts/  tests/  app.py
```

## Limitations

- Trained on LiTS (contrast-enhanced, mostly portal-venous). Performance on non-contrast CT, arterial-phase-only scans, MRI or other populations is unknown without external validation.
- Small lesions (< 10 mm) remain hard. `small_lesion_sensitivity` is reported separately for this reason.
- No Couinaud segmentation and no vessel model, so future liver remnant (FLR) calculation still needs the planned resection plane.
- Synthetic-phantom results (CI) only show that the code works. They say nothing about clinical performance.

## Citation

If you use LiTS, cite: Bilic P. et al., *The Liver Tumor Segmentation Benchmark (LiTS)*, Medical Image Analysis 84 (2023) 102680.

## License

MIT
