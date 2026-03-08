# Pore Analysis AI

Проект для поиска и измерения пор на SEM-изображениях керамики.

Основной рабочий путь в репозитории сейчас такой:

- ручная разметка реальных изображений,
- подготовка корректного `train/val/test`,
- дообучение regression-модели,
- оценка на отложенном реальном тесте.

В репозитории есть две ветки моделей:

- `segmentation`: предсказывает бинарную маску пор;
- `regression`: предсказывает distance map, по которому потом восстанавливаются центры и радиусы пор.

Рекомендуемый вариант для практической работы здесь: `regression`.

## Структура проекта

Что где лежит:

- `tools/annotator/` — GUI-инструмент для ручной разметки.
- `scripts/data_prep/` — подготовка и генерация датасетов.
- `scripts/evaluation/` — оценка модели и анализ результатов.
- `models/regression/` — regression-модель, обучение и inference.
- `models/segmentation/` — segmentation-модель.
- `run/` — готовые команды по шагам, в правильном порядке.
- `RealPoresImages/` — реальные SEM-изображения и ручная разметка.
- `dataset_manual_prepared/` — подготовленный real dataset со split-ами.
- `artifacts/checkpoints/` — все обученные модели и checkpoint-файлы.
- `artifacts/evaluations/` — отчеты оценки и визуализации.
- `artifacts/generated/` — сгенерированные synthetic изображения.

Если смотреть только на порядок запуска, то ориентироваться нужно прежде всего на папку `run/`.

## Быстрый порядок запуска

Самый понятный порядок такой:

1. Разметить реальные изображения.
2. Собрать `train/val/test`.
3. При необходимости обучить модель на synthetic данных.
4. Дообучить модель на реальных данных.
5. Проверить качество на `val`.
6. Проверить итоговое качество на `test`.
7. Проверить устойчивость на `test_robust`.

Готовые команды лежат в:

- `run/01_prepare_real_dataset.sh`
- `run/02_train_regression_on_synthetic.sh`
- `run/03_finetune_regression_on_real.sh`
- `run/04_validate_regression.sh`
- `run/05_test_regression.sh`
- `run/06_test_regression_robust.sh`
- `run/run_all_real_pipeline.sh`

Запускать их можно так:

```bash
bash run/01_prepare_real_dataset.sh
```

## Установка

Создать и активировать окружение:

```bash
python -m venv venv
source venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Если используется NVIDIA GPU, нужен PyTorch с CUDA.

## Шаг 1. Ручная разметка

Запуск аннотатора:

```bash
python tools/annotator/main.py
```

Каждое размеченное изображение сохраняется так:

```text
RealPoresImages/dataset_manual/<sample_id>/
  original.png
  mask.png
  distance_map.png
```

Что это значит:

- `original.png` — исходное изображение;
- `mask.png` — бинарная маска пор;
- `distance_map.png` — карта расстояний для regression-модели.

Соглашение по маске:

- поры — белые (`255`),
- фон — черный (`0`).

## Шаг 2. Подготовка real dataset

Главное правило: сначала делим исходные размеченные изображения на `train/val/test`, и только потом делаем аугментации для `train`.

Запуск:

```bash
python scripts/data_prep/prepare_manual_dataset.py \
  --input RealPoresImages/dataset_manual \
  --output dataset_manual_prepared \
  --train-ratio 0.67 \
  --val-ratio 0.16 \
  --test-ratio 0.17 \
  --with-test-robustness
```

Результат:

```text
dataset_manual_prepared/
  metadata.json
  train/
  val/
  test/
  test_robust/
```

Смысл папок:

- `train/` — только тренировочные изображения и их аугментации;
- `val/` — отдельные изображения для подбора threshold;
- `test/` — отдельные изображения для итоговой проверки;
- `test_robust/` — искаженные версии тестовых изображений для stress-test.

## Шаг 3. Synthetic данные

Если нужен synthetic pretraining:

```bash
python scripts/data_prep/generate_dataset.py --total 5000 --batch-size 100
python prepare_dataset.py
```

Для regression distance maps:

```bash
python models/regression/generate_distance_dataset.py
```

Это создает `dataset_regression/`.

## Шаг 4. Обучение regression-модели

Обучение на synthetic данных:

```bash
python models/regression/train.py \
  --dataset dataset_regression \
  --checkpoint-dir artifacts/checkpoints/regression/synthetic \
  --epochs 50 \
  --batch-size 8
```

Дообучение на реальных данных от уже обученного synthetic checkpoint:

```bash
python models/regression/train.py \
  --dataset dataset_manual_prepared \
  --checkpoint-dir artifacts/checkpoints/regression/real_finetuned \
  --init-model artifacts/checkpoints/regression/synthetic/best_model.pth \
  --epochs 25 \
  --batch-size 4 \
  --learning-rate 1e-4
```

Что делает trainer:

- читает `train`, `val` и при наличии `test`;
- сохраняет `best_model.pth` и `last_model.pth`;
- пишет `history.json` и `training_history.png`.

## Шаг 5. Оценка качества

Оценивать модель нужно так:

1. на `val` выбрать threshold;
2. тот же threshold зафиксировать;
3. только после этого считать итоговую точность на `test`.

Валидация:

```bash
python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real_finetuned/best_model.pth \
  --dataset dataset_manual_prepared/val \
  --output artifacts/evaluations/regression/manual_val_finetuned \
  --thresholds 0.5,1,2,3,4
```

Финальный тест:

```bash
python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real_finetuned/best_model.pth \
  --dataset dataset_manual_prepared/test \
  --output artifacts/evaluations/regression/manual_test_finetuned \
  --fixed-threshold 0.5 \
  --thresholds 0.5,1,2,3,4
```

Проверка устойчивости:

```bash
python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real_finetuned/best_model.pth \
  --dataset dataset_manual_prepared/test_robust \
  --output artifacts/evaluations/regression/manual_test_robust_finetuned \
  --fixed-threshold 0.5 \
  --thresholds 0.5,1,2,3,4
```

Что считает evaluator:

- `Dice` и `IoU` по маске;
- ошибку по пористости;
- `precision / recall / F1` по отдельным порам;
- ошибку по количеству пор;
- ошибку по центрам и радиусам.

## Итоговая точность текущей модели

Текущая финальная модель:

- checkpoint: `artifacts/checkpoints/regression/real_finetuned/best_model.pth`
- стартовала от synthetic pretrained checkpoint;
- дообучалась на 4 размеченных реальных изображениях;
- threshold фиксировался по `val` и потом использовался на `test`.

Как определялась итоговая точность простым языком:

- сначала модель обучили на synthetic данных;
- потом дообучили на нескольких реальных размеченных изображениях;
- одно реальное изображение отложили под `val` и использовали только для выбора threshold;
- другое реальное изображение полностью отложили под финальный `test` и не использовали в обучении;
- дополнительно взяли 8 искаженных версий этого же тестового изображения и проверили устойчивость модели.

Итог на реальном `test`:

- `Dice = 0.9709`
- `IoU = 0.9435`
- предсказанная пористость: `6.31%`
- истинная пористость: `6.54%`
- ошибка по пористости: `0.24` процентного пункта
- найдено пор: `36`
- истинно пор: `57`
- `pore recall = 0.6316`
- `pore precision = 1.0000`
- `pore F1 = 0.7742`
- `radius MAE = 0.1509 px`

Итог на `test_robust`:

- средний `Dice = 0.9658`
- средний `IoU = 0.9339`
- средняя ошибка по пористости: `0.28` процентного пункта
- в среднем найдено пор: `35`
- истинно пор: `57`
- средний `pore recall = 0.6140`
- средний `pore precision = 1.0000`
- средний `pore F1 = 0.7607`

Что это означает нормальным языком:

- модель очень хорошо восстанавливает общую площадь пор;
- оценка общей пористости уже близка к ручной разметке;
- лишние поры модель почти не придумывает;
- главный текущий недостаток — недосчет отдельных пор, когда они расположены плотно или плохо разделяются.

Важно: эти цифры пока предварительные, потому что реальный тестовый набор еще маленький.

## Segmentation ветка

Она остается в репозитории, но основной рабочий путь сейчас не через нее:

```bash
python models/segmentation/train.py
python models/segmentation/inference.py --model checkpoints/best_model.pth --dataset dataset
```

## Проверка

Unit-тесты:

```bash
pytest
```

Интеграционный тест запускается отдельно:

```bash
python tests/test_integration.py
```
