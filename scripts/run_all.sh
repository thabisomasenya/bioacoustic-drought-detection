#!/usr/bin/env bash
# Full, resumable reproduction of every result in the report.
# Each step is skipped if its output already exists, and tuning/CV resume mid-way,
# so the script can simply be re-run after any interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
CLI="python -m drought_acoustics.cli"
EXP=configs/experiments
BEST=artifacts/tuning/cnn1d_aug/best_params.json

has_run() { for d in runs/*_"$1"; do [ -f "$d/metrics.json" ] && return 0; done; return 1; }
has_cv()  { for d in runs/*_"$1"_cv; do [ -f "$d/cv_summary.json" ] && return 0; done; return 1; }

# 1. data audit + exploratory analysis
[ -f artifacts/manifest.csv ]        || $CLI manifest
[ -f artifacts/eda/04_mean_psd.png ] || $CLI eda

# 2. hyperparameter search for the proposed model (validation split only)
[ -f "$BEST" ] || $CLI tune -e $EXP/cnn1d_aug.yaml

# 3. holdout runs: proposed model, augmentation ablation, Deliverable 1 baselines
has_run cnn1d_aug     || $CLI train -e $EXP/cnn1d_aug.yaml   --params "$BEST"
has_run cnn1d_noaug   || $CLI train -e $EXP/cnn1d_noaug.yaml --params "$BEST"
has_run cnn2d_mel_aug || $CLI train -e $EXP/cnn2d_mel_aug.yaml
has_run lstm_mel_aug  || $CLI train -e $EXP/lstm_mel_aug.yaml

# 4. 5-fold plant-grouped CV on identical folds (paired model comparison)
has_cv cnn1d_aug     || $CLI train -e $EXP/cnn1d_aug.yaml     --params "$BEST" --cv --verbose 0
has_cv cnn2d_mel_aug || $CLI train -e $EXP/cnn2d_mel_aug.yaml --cv --verbose 0
has_cv lstm_mel_aug  || $CLI train -e $EXP/lstm_mel_aug.yaml  --cv --verbose 0
has_cv cnn1d_noaug   || $CLI train -e $EXP/cnn1d_noaug.yaml   --params "$BEST" --cv --verbose 0

# 5. comparison tables and figures
$CLI compare
echo "ALL DONE"
