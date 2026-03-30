# Структура репозитория

Каноническое дерево и назначение каталогов/файлов. Пути относительно корня репозитория.

## Дерево

```
.
├── app/                          # Устанавливаемый Python-пакет (импорт: app.*)
│   ├── api/
│   │   ├── main.py               # FastAPI: корень, middleware, v1 router
│   │   ├── deps.py               # get_inference_service
│   │   ├── errors.py             # Единый JSON ошибок + X-Request-ID
│   │   ├── middleware.py         # Логирование запросов, request id
│   │   └── v1/routes.py          # /v1/health, model-info, predict, predict_batch
│   ├── core/
│   │   ├── config.py             # YAML + pydantic-settings (.env)
│   │   ├── logger.py
│   │   └── schemas.py            # Pydantic-модели API
│   ├── data/
│   │   ├── loader.py             # Чтение CSV
│   │   ├── schema.py             # Контракт колонок, валидация «сырья»
│   │   ├── pipeline.py           # Оркестрация: load → validate → профиль
│   │   ├── preprocessing.py      # Sklearn ColumnTransformer, fit/transform, joblib
│   │   ├── split.py              # Stratified train/val/test
│   │   └── dataset.py            # torch.utils.data.Dataset
│   ├── features/
│   │   ├── build_features.py     # Списки признаков, target, id (single source of truth)
│   │   └── ablation.py            # Наборы колонок для ablation-экспериментов
│   ├── models/
│   │   ├── net.py                # TabularMLP
│   │   ├── train.py              # Цикл обучения, early stopping, артефакты
│   │   ├── evaluate.py           # Метрики, confusion matrix
│   │   ├── baselines.py          # LogReg, RF, XGBoost на матрице признаков
│   │   ├── benchmark.py          # Ablation × baselines (+ опц. короткий MLP)
│   │   ├── predict.py            # Инференс batch/one
│   │   └── utils.py              # seed, device
│   ├── services/
│   │   ├── inference_service.py  # Загрузка артефактов, predict для API/UI
│   │   ├── data_service.py       # Демо-данные, список CSV, превью, валидация сырья
│   │   ├── training_service.py # Полный цикл обучения (оркестрация models + data)
│   │   ├── evaluation_service.py # Метрики, отчёты в artifacts/
│   │   ├── prediction_service.py # Одиночный и batch predict для UI
│   │   ├── status_service.py     # Статус модели, обзор проекта, артефакты
│   │   └── config_helpers.py     # Подстановка путей и гиперпараметров из UI
│   ├── ui/
│   │   ├── streamlit_app.py      # Точка входа: дипломная навигация по этапам исследования
│   │   ├── demo_components.py    # Стили, заголовок страницы + ссылка на defense_narrative.md
│   │   ├── prediction_demo_config.py  # Сценарии прогноза и условные веса признаков
│   │   ├── russian_ui.py         # Русские подписи признаков и метрик для UI
│   │   ├── utils.py              # Загрузка конфига для UI
│   │   └── views/                # project_about, problem_statement, data, preprocessing,
│   │                             # training, evaluation, prediction_demo, practical_value, help
│   └── utils/
│       └── common.py             # Корень проекта, resolve_path
├── artifacts/                    # Не коммитить обученные веса в монорепо при политике LFS
│   ├── models/
│   ├── metrics/
│   ├── reports/
│   └── encoders/
├── configs/
│   └── config.yaml               # Единый конфиг эксперимента
├── data/
│   ├── raw/                      # Вход: trips_demo.csv и т.п.
│   ├── processed/                # test_split.csv, профили валидации
│   └── external/                 # Зарезервировано под внешние справочники
├── docs/                         # ВКР, 01–09, PROJECT_GUIDE, PROJECT_STRUCTURE, defense_*
├── examples/                     # JSON для API/CLI
├── notebooks/
│   └── eda.ipynb
├── scripts/
│   ├── generate_demo_data.py
│   ├── prepare_data.py           # Валидация и профиль сырья
│   ├── compare_baselines.py      # Baselines + ablations, отчёт в artifacts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── launch_ui.py              # Обёртка: streamlit run app/ui/streamlit_app.py
├── tests/                        # pytest
├── Dockerfile                    # Образ FastAPI (API)
├── Dockerfile.ui                 # Образ Streamlit (UI)
├── docker-compose.yml            # По умолчанию UI; профиль api — поднять API
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Границы ответственности

| Слой | Где живёт | Запрещено |
|------|-----------|-----------|
| Контракт данных | `app/features/build_features.py`, `app/data/schema.py` | Дублировать списки колонок в скриптах вручную |
| Препроцессинг | `app/data/preprocessing.py` | Fit на val/test; разные правила train vs inference |
| Обучение | `app/models/train.py`, `app/services/training_service.py`, `scripts/train.py` | Смешивать оценку на train как финальную метрику |
| Сервис | `app/services/*.py` | Дублировать цикл обучения или препроцессинг в `app/ui/` |
| UI | `app/ui/` | Вызывать `app/models/train.py` напрямую вместо сервисов |
| API | `app/api/` | Обходить `InferenceService` при предсказаниях |

## Точки расширения

- Новые признаки: обновить `build_features.py`, `schema.py`, генератор `scripts/generate_demo_data.py`, Pydantic `app/core/schemas.py`.
- Новый источник данных: реализовать читатель рядом с `loader.py`, подключить в `pipeline.py`.
