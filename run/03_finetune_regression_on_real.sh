#!/usr/bin/env bash
set -euo pipefail

python models/regression/train.py \
  --dataset dataset_manual_prepared \
  --checkpoint-dir artifacts/checkpoints/regression/real_finetuned \
  --init-model artifacts/checkpoints/regression/synthetic/best_model.pth \
  --epochs 25 \
  --batch-size 4 \
  --learning-rate 1e-4
