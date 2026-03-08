#!/usr/bin/env bash
set -euo pipefail

python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real/best_model.pth \
  --dataset dataset_manual_prepared/test_robust \
  --output artifacts/evaluations/regression/manual_test_robust_real \
  --fixed-threshold 0.5 \
  --thresholds 0.5,1,2,3,4
