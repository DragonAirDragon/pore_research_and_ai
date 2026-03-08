# Pore Analysis AI

Проект для поиска и измерения пор на SEM-изображениях керамики.

В упрощенном виде проект работает так:

1. Берет реальные изображения материала.
2. На нескольких изображениях человек вручную размечает поры.
3. Из этих размеченных данных собирается датасет `train / val / test`.
4. Модель учится находить поры.
5. Потом мы проверяем, насколько хорошо она работает на изображениях, которые не показывали ей во время обучения.

Основной рабочий путь в этом репозитории сейчас только один: обучение и тестирование на реальных данных и на их аугментациях.

## Что здесь главное

Если нужен короткий ориентир по репозиторию, достаточно помнить следующее:

- основной тип модели: `regression`
- основная папка с командами: `run/`
- основной подготовленный датасет: `dataset_manual_prepared/`
- основные результаты модели: `artifacts/`

## Что лежит в папках

- `tools/annotator/` — программа для ручной разметки пор.
- `scripts/data_prep/` — подготовка датасета и аугментации.
- `scripts/evaluation/` — расчет метрик и оценка модели.
- `models/regression/` — основная модель, которую нужно запускать.
- `models/segmentation/` — альтернативная ветка модели. Она есть в репозитории, но основной workflow сейчас не через нее.
- `run/` — готовые команды по порядку, чтобы не вспоминать вручную, что запускать.
- `RealPoresImages/` — реальные изображения и ручная разметка.
- `dataset_manual_prepared/` — уже собранный датасет со split-ами `train / val / test / test_robust`.
- `artifacts/checkpoints/` — сохраненные веса модели.
- `artifacts/evaluations/` — итоговые отчеты, метрики и результаты оценки.

## Быстрый запуск

Есть два сценария.

### Сценарий 1. У тебя уже есть готовый датасет

Тогда первый шаг с ручной разметкой можно пропустить.

Запускай так:

```bash
bash run/03_train_regression_on_real.sh
bash run/04_validate_regression.sh
bash run/05_test_regression.sh
bash run/06_test_regression_robust.sh
```

### Сценарий 2. Ты хочешь пройти путь с самого начала

Запускай так:

```bash
bash run/01_prepare_real_dataset.sh
bash run/03_train_regression_on_real.sh
bash run/04_validate_regression.sh
bash run/05_test_regression.sh
bash run/06_test_regression_robust.sh
```

Или одной командой:

```bash
bash run/run_all_real_pipeline.sh
```

## Готовый датасет

Если нет необходимости заново собирать и размечать данные, можно использовать уже готовый датасет:

[Google Drive: готовый датасет](https://drive.google.com/file/d/1D3RilH6dyAzNCyiDUmFHEfnJkgg7rz6o/view?usp=sharing)

### Как установить датасет

1. Скачайте архив по ссылке выше.
2. Распакуйте его.
3. Поместите содержимое архива в папку проекта.
4. После распаковки в папке проекта должны появиться или обновиться, например, такие папки:

  `RealPoresImages/`
  `dataset_manual_prepared/`
  `artifacts/`
5. После этого можно сразу переходить к обучению или тестированию без повторной ручной разметки.

Если после распаковки `dataset_manual_prepared/` уже есть, то шаг подготовки датасета можно не запускать.

## Установка окружения

Создать виртуальное окружение:

```bash
python -m venv venv
```

Активировать его:

```bash
source venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

Если у тебя NVIDIA GPU, желательно использовать PyTorch с CUDA. Если GPU нет, проект можно запускать и на CPU, просто медленнее.

## Шаг 1. Ручная разметка

Если размечать данные вручную, запускать аннотатор нужно так:

```bash
python tools/annotator/main.py
```

Каждое размеченное изображение сохраняется в таком виде:

```text
RealPoresImages/dataset_manual/<sample_id>/
  original.png
  mask.png
  distance_map.png
```

Что это значит:

- `original.png` — исходное изображение.
- `mask.png` — картинка, где поры отмечены белым, а фон черным.
- `distance_map.png` — вспомогательная карта расстояний, по которой regression-модель учится лучше выделять поры.

Кратко: `mask.png` показывает, где находятся поры, а `distance_map.png` помогает модели точнее понять их форму и центр.

## Шаг 2. Подготовка датасета

После разметки нужно собрать датасет для обучения и теста.

Главное правило здесь такое:

- сначала делим изображения на `train / val / test`
- только потом делаем аугментации для `train`

Это важно, чтобы модель не увидела почти одинаковые картинки и чтобы оценка была честной.

Команда:

```bash
python scripts/data_prep/prepare_manual_dataset.py \
  --input RealPoresImages/dataset_manual \
  --output dataset_manual_prepared \
  --train-ratio 0.67 \
  --val-ratio 0.16 \
  --test-ratio 0.17 \
  --with-test-robustness
```

Что получится:

```text
dataset_manual_prepared/
  metadata.json
  train/
  val/
  test/
  test_robust/
```

Что значат эти папки:

- `train/` — на этих данных модель учится.
- `val/` — на этих данных подбираются параметры оценки, например threshold.
- `test/` — на этих данных считается финальная честная точность.
- `test_robust/` — это тот же тест, но с ухудшениями изображения: шум, яркость, контраст и т.д. Нужен для проверки устойчивости.

## Шаг 3. Обучение модели

Основная модель здесь — regression-модель.

Она не просто рисует маску, а предсказывает `distance map`, из которой потом восстанавливаются центры и радиусы пор.

Этот подход особенно удобен, когда поры расположены близко друг к другу и обычная бинарная маска может объединять их в одну область.

Команда обучения:

```bash
python models/regression/train.py \
  --dataset dataset_manual_prepared \
  --checkpoint-dir artifacts/checkpoints/regression/real \
  --epochs 25 \
  --batch-size 4 \
  --learning-rate 1e-4
```

Что сохраняется после обучения:

- `best_model.pth` — лучшая версия модели.
- `last_model.pth` — последняя версия модели.
- `history.json` — история обучения по эпохам.
- `training_history.png` — график обучения.

## Шаг 4. Проверка качества

Проверка идет в три этапа:

1. Сначала смотрим `val`.
2. Потом фиксируем threshold.
3. Только после этого считаем финальную метрику на `test`.

Это нужно, чтобы не подгонять результат под тест.

### Проверка на `val`

```bash
python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real/best_model.pth \
  --dataset dataset_manual_prepared/val \
  --output artifacts/evaluations/regression/manual_val_real \
  --thresholds 0.5,1,2,3,4
```

### Финальный тест на `test`

```bash
python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real/best_model.pth \
  --dataset dataset_manual_prepared/test \
  --output artifacts/evaluations/regression/manual_test_real \
  --fixed-threshold 0.5 \
  --thresholds 0.5,1,2,3,4
```

### Проверка устойчивости на `test_robust`

```bash
python scripts/evaluation/evaluate_real_dataset.py \
  --task regression \
  --model artifacts/checkpoints/regression/real/best_model.pth \
  --dataset dataset_manual_prepared/test_robust \
  --output artifacts/evaluations/regression/manual_test_robust_real \
  --fixed-threshold 0.5 \
  --thresholds 0.5,1,2,3,4
```

## Что значат метрики

Ниже объяснение простым языком.

- `Dice` — насколько хорошо совпали предсказанная маска и ручная разметка. Чем ближе к `1.0`, тем лучше.
- `IoU` — тоже показывает совпадение масок, но считается чуть строже. Чем ближе к `1.0`, тем лучше.
- `precision` — сколько найденных моделью пор действительно являются порами. Высокий precision означает, что модель почти не отмечает лишние поры.
- `recall` — сколько настоящих пор модель смогла найти. Высокий recall значит, что модель мало пропускает.
- `F1` — компромисс между precision и recall. Хорошая общая оценка качества поиска отдельных пор.
- `porosity` — доля площади изображения, которая занята порами.
- `porosity error` — насколько модель ошиблась в оценке общей пористости.
- `count_pred` — сколько пор модель нашла.
- `count_gt` — сколько пор есть в ручной разметке.
- `count_error` — насколько модель ошиблась по количеству пор.
- `center_mae` — средняя ошибка положения центра поры. Чем меньше, тем лучше.
- `radius_mae` — средняя ошибка радиуса поры. Чем меньше, тем лучше.
- `distance_mae` и `distance_rmse` — ошибки самой distance map. Это технические метрики качества регрессии. Они полезны, но для человека обычно важнее Dice, IoU, porosity error, precision, recall и count error.

Если нужен совсем краткий ориентир:

- хочешь понять, насколько совпала маска: смотри `Dice` и `IoU`
- если нужно понять, насколько точна оценка общей пористости: смотри `porosity error`
- хочешь понять, насколько хорошо модель находит отдельные поры: смотри `precision`, `recall`, `F1`, `count_error`

## Итоговый результат текущей модели

Важно: для этого этапа были вручную размечены только 6 реальных изображений пор. Это очень маленький объем данных. Несмотря на это, модель уже показала отличный результат.

Как считалась итоговая точность:

1. Были вручную размечены 6 реальных изображений.
2. Из них собрали `train / val / test`.
3. Для `train` сделали аугментации.
4. Модель обучили только на `train`.
5. На `val` выбрали threshold.
6. На `test` посчитали финальную точность.
7. На `test_robust` проверили устойчивость к ухудшению качества изображения.

### Результат на `test`

- `Dice = 0.9709`
  Это очень хорошее совпадение маски с ручной разметкой.
- `IoU = 0.9435`
  Это тоже очень сильный результат.
- предсказанная пористость: `6.31%`
- истинная пористость: `6.54%`
- ошибка по пористости: `0.24` процентного пункта
  То есть общую площадь пор модель оценивает очень близко к реальности.
- найдено пор: `36`
- истинно пор: `57`
- `pore recall = 0.6316`
  Модель находит не все поры, часть пропускает.
- `pore precision = 1.0000`
  Это означает, что модель почти не отмечает лишние поры.
- `pore F1 = 0.7742`
  В целом качество поиска отдельных пор уже хорошее, но есть запас для улучшения.
- `radius MAE = 0.1509 px`
  Ошибка по радиусу маленькая.

### Результат на `test_robust`

- средний `Dice = 0.9658`
- средний `IoU = 0.9339`
- средняя ошибка по пористости: `0.28` процентного пункта
- в среднем найдено пор: `35`
- истинно пор: `57`
- средний `pore recall = 0.6140`
- средний `pore precision = 1.0000`
- средний `pore F1 = 0.7607`

Это значит, что даже после ухудшения изображения модель все равно работает стабильно.

## Что можно сказать о модели простым языком

Сильные стороны:

- очень хорошо определяет общую площадь пор
- очень близко оценивает пористость
- почти не рисует лишние поры
- достаточно устойчиво работает даже на ухудшенных изображениях

Слабое место сейчас одно:

- модель недосчитывает часть отдельных пор, особенно когда поры расположены плотно или плохо отделяются друг от друга

На текущем этапе модель уже хорошо подходит для оценки общей пористости и общей структуры пор, однако для максимально точного подсчета каждой отдельной поры данных пока недостаточно.

## Какие команды реально нужны каждый день

Если оставить только самое нужное:

```bash
python tools/annotator/main.py
python scripts/data_prep/prepare_manual_dataset.py --input RealPoresImages/dataset_manual --output dataset_manual_prepared --train-ratio 0.67 --val-ratio 0.16 --test-ratio 0.17 --with-test-robustness
python models/regression/train.py --dataset dataset_manual_prepared --checkpoint-dir artifacts/checkpoints/regression/real --epochs 25 --batch-size 4 --learning-rate 1e-4
python scripts/evaluation/evaluate_real_dataset.py --task regression --model artifacts/checkpoints/regression/real/best_model.pth --dataset dataset_manual_prepared/test --output artifacts/evaluations/regression/manual_test_real --fixed-threshold 0.5 --thresholds 0.5,1,2,3,4
```

## Проверка

Unit-тесты:

```bash
pytest
```

