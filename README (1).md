# Liver Tumor Segmentation from CT using 3D U-Net

Automatic segmentation of liver and liver tumors from contrast-enhanced abdominal CT scans, using a 3D U-Net trained on the LiTS (Liver Tumor Segmentation) dataset.

## Motivation & Clinical Relevance

Accurate liver and tumor segmentation is a prerequisite for several decisions in HPB surgical oncology:

- **Future Liver Remnant (FLR) estimation** before major hepatectomy — underestimating remnant volume risks post-hepatectomy liver failure.
- **Tumor burden quantification** for treatment planning (resection vs. ablation vs. systemic therapy).
- **Longitudinal tracking** of lesion size/number across follow-up imaging.

Manual segmentation is time-consuming and suffers from inter-observer variability, particularly for tumor boundaries. This project explores whether a 3D U-Net can approximate expert-level segmentation to support (not replace) radiological and surgical decision-making.

## Dataset

This project primarily uses **LiTS**, with optional external validation and extension datasets listed below. All datasets are free, multi-institutional, and de-identified.

### Primary: LiTS (Liver Tumor Segmentation Benchmark)

- **Source:** [The Liver Tumor Segmentation Benchmark](https://competitions.codalab.org/competitions/17094) — data pooled from 7 hospitals and research institutions
- **Mirror:** Kaggle (`andrewmvd/liver-tumor-segmentation`)
- **Modality:** Contrast-enhanced CT
- **Format:** NIfTI (`.nii.gz`)
- **Labels:** 0 = background, 1 = liver, 2 = tumor
- **Size:** 131 labeled CT volumes (training), 70 unlabeled (official test set)
- **Citation:** Bilic et al., "The Liver Tumor Segmentation Benchmark (LiTS)," *Medical Image Analysis*, 2023.

Dataset is **not included** in this repository due to size and licensing. See [Setup](#setup) for download instructions.

### Optional: External Validation & Extensions

| Dataset | Source | Modality / Format | Use in this project |
|---|---|---|---|
| **3D-IRCADb-01** | [IRCAD, Strasbourg](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) — 20 patients, multiple European hospitals, various scanners | CT, DICOM/VTK (NIfTI conversions available on GitHub/Dropbox mirrors) | External validation set — tests generalization beyond LiTS's source hospitals. Also includes hepatic vessel and Couinaud segment annotations. |
| **MSD Task08 – Hepatic Vessel** | [Medical Segmentation Decathlon](http://medicaldecathlon.com/), Memorial Sloan Kettering Cancer Center | CT, NIfTI | Extension task: hepatic vessel + tumor segmentation, directly relevant to surgical planning (e.g. vascular relationship to resection margins). |
| **TCGA-LIHC** | [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/tcga-lihc/), multi-center via TCGA | CT/MRI (DICOM) + clinical + histopathology (WSI) | For future multimodal work — combining imaging with clinical/pathological data. Requires free TCIA account. |

## Downloading the Datasets

**LiTS (Kaggle):**
```bash
kaggle datasets download -d andrewmvd/liver-tumor-segmentation
unzip liver-tumor-segmentation.zip -d data/raw/
```

**3D-IRCADb-01 (direct, no account needed):**
Download individual patient folders or the full set from the IRCAD dataset page. Files are provided in DICOM; convert to NIfTI using `dicom2nifti` or `SimpleITK` before running `preprocessing.py`.

**MSD Task08 (Hepatic Vessel):**
```bash
# via Hugging Face mirror
wget https://huggingface.co/datasets/Novel-BioMedAI/Medical_Segmentation_Decathlon/resolve/main/Task08_HepaticVessel.tar
tar -xf Task08_HepaticVessel.tar
```

**TCGA-LIHC (requires free TCIA account):**
1. Create an account at cancerimagingarchive.net
2. Use the NBIA Data Retriever tool to download the TCGA-LIHC collection
3. Clinical data is available separately via the GDC Data Portal

> Note: file naming conventions differ across datasets. `preprocessing.py` is written for LiTS's `volume-X.nii.gz` / `segmentation-X.nii.gz` convention — adjust `get_patient_ids()` accordingly for other sources.

## Pipeline Overview

```
Raw NIfTI (CT + segmentation)
        ↓
HU Windowing (liver window: level=50, width=400)
        ↓
Resampling to isotropic spacing (1.0 × 1.0 × 1.5 mm)
        ↓
Patient-level train/val/test split (70/15/15)
        ↓
3D U-Net training (patch-based, 128×128×64)
        ↓
Evaluation: Dice score, separately for liver and tumor
```

## Repository Structure

```
liver-tumor-segmentation/
├── README.md
├── requirements.txt
├── data/                      # not tracked in git — see Setup
│   ├── raw/
│   ├── train/{images,masks}/
│   ├── val/{images,masks}/
│   └── test/{images,masks}/
├── src/
│   ├── preprocessing.py       # windowing, resampling, patient-level split
│   ├── model.py                # 3D U-Net architecture
│   ├── losses.py               # Dice + weighted Cross-Entropy loss
│   ├── train.py                 # training loop with checkpointing
│   └── evaluate.py             # test-set evaluation
├── notebooks/
│   └── data_exploration.ipynb
└── outputs/
    ├── checkpoints/
    └── predictions/
```

## Setup

```bash
git clone https://github.com/<your-username>/liver-tumor-segmentation.git
cd liver-tumor-segmentation
pip install -r requirements.txt
```

Download the LiTS dataset and place volumes/masks in `data/raw/` as:

```
data/raw/volume-0.nii.gz
data/raw/segmentation-0.nii.gz
data/raw/volume-1.nii.gz
...
```

Run preprocessing:

```bash
python src/preprocessing.py
```

This will populate `data/train/`, `data/val/`, and `data/test/` with preprocessed `.npy` volumes, split **by patient** to prevent data leakage.

## Training

```bash
python src/train.py
```

Key settings (in `src/train.py`):

| Parameter | Value | Notes |
|---|---|---|
| Patch size | 64×128×128 | Memory-constrained; adjust to GPU |
| Batch size | 2 | Increase if GPU memory allows |
| Optimizer | Adam | lr=1e-4, ReduceLROnPlateau scheduler |
| Loss | Dice + weighted Cross-Entropy | Tumor class up-weighted (0.5) due to class imbalance |
| Early stopping | 20 epochs patience | Based on validation loss |

## Evaluation

Model performance is reported **separately for liver and tumor**, since tumor segmentation is substantially harder due to smaller size and irregular boundaries:

| Class | Dice Score (val) |
|---|---|
| Liver | *fill in after training* |
| Tumor | *fill in after training* |

> A single averaged Dice score across all classes is misleading here — liver Dice is typically 0.90+ while tumor Dice is commonly in the 0.60–0.75 range even for strong models, due to smaller lesion size and boundary ambiguity.

## Example Predictions

*(To be added after training — include 2–3 axial slices showing CT, ground truth mask, and predicted mask side by side.)*

## Limitations

- Trained on a single-center dataset (LiTS); generalization to other scanners/protocols is untested.
- Small tumors (<10mm) are known to be harder to segment reliably; performance likely degrades on these cases.
- No external validation set was used; reported metrics reflect internal train/val/test split only.
- 3D patch-based training may miss global context available with full-volume inference.

## Future Work

- Compare against nnU-Net (self-configuring framework, current standard for medical segmentation benchmarks).
- Add post-processing (largest connected component filtering) to reduce false positives.
- External validation on an independent cohort.
- Extend to multi-task setup predicting tumor grade alongside segmentation.

## Citation

If you use the LiTS dataset, please cite the original challenge paper (see dataset page for current citation).

## License

MIT License (or specify your preferred license).
