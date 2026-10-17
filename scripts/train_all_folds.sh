#!/usr/bin/env bash
# Full recipe: preprocess -> 5-fold training -> ensemble evaluation on the held-out test set.
# Usage: bash scripts/train_all_folds.sh configs/lits_3d_fullres.yaml
set -euo pipefail
CFG=${1:-configs/lits_3d_fullres.yaml}
EXP=$(python -c "from liverseg.config import load_config; print(load_config('$CFG')['experiment'])")
python -m liverseg.preprocessing --config "$CFG" --num-workers 8
for FOLD in 0 1 2 3 4; do
  python -m liverseg.train --config "$CFG" --fold $FOLD --resume
done
python -m liverseg.evaluate --config "$CFG" --model "outputs/$EXP" --out "results/$EXP"
