#!/usr/bin/env bash
set -euo pipefail

python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real_finetuned/best_model.pth \
  --dataset dataset_manual_prepared/val \
  --output artifacts/evaluations/regression/manual_val_finetuned \
  --thresholds 0.5,1,2,3,4
