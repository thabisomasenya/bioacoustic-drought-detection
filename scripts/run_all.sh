#!/usr/bin/env bash
# Full, resumable reproduction of every result in the report.
# Each step is skipped if its output already exists, so the script can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
CLI="python -m drought_acoustics.cli"
EXP=configs/experiments
BEST=artifacts/tuning/cnn1d_aug/best_params.json

has_run() { ls -d runs/*_"$1" 2>/dev/null | while read -r d; do [ -f "$d/metrics.json" ] && echo yes; done | grep -q yes; }

[ -f artifacts/manifest.csv ]          || $CLI manifest
[ -f artifacts/eda/04_mean_psd.png ]   || $CLI eda
[ -f "$BEST" ]                         || $CLI tune -e $EXP/cnn1d_aug.yaml

has_run cnn1d_aug     || $CLI train -e $EXP/cnn1d_aug.yaml   --params "$BEST" --verbose 2
has_run cnn1d_noaug   || $CLI train -e $EXP/cnn1d_noaug.yaml --params "$BEST" --verbose 2
has_run cnn2d_mel_aug || $CLI train -e $EXP/cnn2d_mel_aug.yaml --verbose 2
has_run lstm_mel_aug  || $CLI train -e $EXP/lstm_mel_aug.yaml  --verbose 2
$CLI compare
ls runs/*_cnn1d_aug_cv/cv_summary.json >/dev/null 2>&1 || $CLI train -e $EXP/cnn1d_aug.yaml --params "$BEST" --cv --verbose 0
echo "ALL DONE"
