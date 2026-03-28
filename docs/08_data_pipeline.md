# Data pipeline

Описание **офлайн-цепочки данных** от сырого CSV до матриц признаков для нейросети. Инференс использует те же правила преобразования через сохранённый `preprocessor.joblib`.

## Этапы

```mermaid
flowchart LR
  A[raw CSV] --> B[load]
  B --> C[validate schema]
  C --> D[optional profile JSON]
  D --> E[train.py: split]
  E --> F[fit preprocessor on train]
  F --> G[transform val/test]
```

1. **Ingest** — `app/data/loader.py` читает CSV в `pandas.DataFrame`.
2. **Validate** — `app/data/schema.py` проверяет наличие колонок, типы/значения таргета, опционально категории (`strict`).
3. **Profile** — `app/data/pipeline.py` пишет агрегаты (число строк, доля пропусков, распределение таргета) в `data/processed/` для аудита.
4. **Split** — `app/data/split.py`, стратификация по `target_class`, фиксированный seed из конфига.
5. **Preprocess** — `app/data/preprocessing.py`: только **train** для `fit`; val/test — `transform`; артефакт на диск.

## Анти-утечки

| Операция | Разрешено на |
|----------|----------------|
| `fit` imputer / scaler / one-hot | train |
| `transform` | train, val, test, inference |
| Подбор порога по «удобству» на test | **Нет** (тест только для финальной отчётности) |

## CLI

```bash
python scripts/prepare_data.py --input data/raw/trips_demo.csv --strict
```

Выход: код 0 при успехе, ненулевой при ошибках валидации. Опция `--write-profile` сохраняет JSON профиля.

## Замена на прод-данные

1. Выгрузить CSV с колонками из `docs/03_data_description.md`.
2. Прогнать `prepare_data.py`.
3. Указать путь в `configs/config.yaml` → `paths.raw_data`.
4. Запустить `scripts/train.py`.

Если категории шире демо — не использовать `--strict` до обновления `CATEGORICAL_ALLOWED` в `app/data/schema.py`.

## Зависимости между модулями

- Единственный источник имён признаков для обучения: `app/features/build_features.py`.
- Машинные ограничения: `app/data/schema.py` (синхронизирован с документацией).
