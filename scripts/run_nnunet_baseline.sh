#!/usr/bin/env bash
# nnU-Net v2 baseline on the same split, scored with our evaluator.
# Requires: pip install nnunetv2 ; a GPU ; nnUNet_raw / nnUNet_preprocessed / nnUNet_results set.
set -euo pipefail
DS=17
python scripts/convert_to_nnunet.py --lits data/raw --out "$nnUNet_raw/Dataset0${DS}_LiTS" \
    --exclude-test data/preprocessed/3d_fullres/splits.json
nnUNetv2_plan_and_preprocess -d $DS --verify_dataset_integrity
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train $DS 3d_fullres $FOLD
done
nnUNetv2_predict -i "$nnUNet_raw/Dataset0${DS}_LiTS/imagesTs" -o results/nnunet_preds -d $DS -c 3d_fullres
liverseg-evaluate --pred-dir results/nnunet_preds --gt-dir "$nnUNet_raw/Dataset0${DS}_LiTS/labelsTs" --out results/nnunet
