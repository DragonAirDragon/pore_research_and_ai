#!/usr/bin/env bash
set -euo pipefail

python models/regression/train.py \
  --dataset dataset_regression \
  --checkpoint-dir artifacts/checkpoints/regression/synthetic \
  --epochs 50 \
  --batch-size 8
