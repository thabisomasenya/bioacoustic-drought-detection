#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CLI="python -m drought_acoustics.cli"; EXP=configs/experiments; BEST=artifacts/tuning/cnn1d_aug/best_params.json
ls runs/*_cnn2d_mel_aug_cv/cv_summary.json >/dev/null 2>&1 || $CLI train -e $EXP/cnn2d_mel_aug.yaml --cv --verbose 0
ls runs/*_lstm_mel_aug_cv/cv_summary.json  >/dev/null 2>&1 || $CLI train -e $EXP/lstm_mel_aug.yaml --cv --verbose 0
ls runs/*_cnn1d_noaug_cv/cv_summary.json   >/dev/null 2>&1 || $CLI train -e $EXP/cnn1d_noaug.yaml --params "$BEST" --cv --verbose 0
$CLI compare
echo "CV DONE"
