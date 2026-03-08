# Run Order

Основной порядок запуска:

1. `bash run/01_prepare_real_dataset.sh`
2. `bash run/02_train_regression_on_synthetic.sh`
3. `bash run/03_finetune_regression_on_real.sh`
4. `bash run/04_validate_regression.sh`
5. `bash run/05_test_regression.sh`
6. `bash run/06_test_regression_robust.sh`
7. Или сразу `bash run/run_all_real_pipeline.sh`

Если synthetic pretraining уже есть, можно начинать с шага 1, затем перейти сразу к шагу 3.
