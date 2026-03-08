#!/usr/bin/env bash
set -euo pipefail

python scripts/data_prep/prepare_manual_dataset.py \
  --input RealPoresImages/dataset_manual \
  --output dataset_manual_prepared \
  --train-ratio 0.67 \
  --val-ratio 0.16 \
  --test-ratio 0.17 \
  --with-test-robustness
