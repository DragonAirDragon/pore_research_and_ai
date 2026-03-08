#!/usr/bin/env bash
set -euo pipefail

bash run/01_prepare_real_dataset.sh
bash run/03_finetune_regression_on_real.sh
bash run/04_validate_regression.sh
bash run/05_test_regression.sh
bash run/06_test_regression_robust.sh
