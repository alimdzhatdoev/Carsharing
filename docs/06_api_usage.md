# API: использование

Сервис: **FastAPI**, по умолчанию `http://127.0.0.1:8000`.

## Версионирование

- **Семантическая версия сервиса** (релиз приложения): поле `service_semantic_version` в ответах и в OpenAPI (`app.version`), текущее значение задаётся в `app/core/api_constants.py` (`SERVICE_SEMANTIC_VERSION`).
- **Версия маршрутов API:** префикс **`/v1`**. Стабильный контракт для клиентов: все операции вызываются как `/v1/...`. Корень **`GET /`** возвращает JSON с актуальным `base_path` и списком endpoint’ов (удобно для интеграций и дипломного описания).

Устаревшие пути без `/v1` **не поддерживаются** — клиенты должны перейти на `/v1`.

## Логирование запросов

- Каждому запросу назначается **`X-Request-ID`** (UUID), если клиент не передал свой заголовок `X-Request-ID`.
- В лог (stdout) пишется строка: `request_id`, метод, путь, HTTP-статус, длительность в мс.
- Ответы (включая ошибки валидации и `HTTPException`) дублируют идентификатор в заголовке **`X-Request-ID`** для трассировки.

## Обработка ошибок

Единый формат тела при ошибках (кроме редких системных ответов Starlette без тела):

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request body or parameters failed validation",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "details": []
  }
}
```

- **`VALIDATION_ERROR`** (422) — тело не прошло Pydantic; в `details` — массив ошибок FastAPI/Pydantic.
- **`SERVICE_UNAVAILABLE`** (503) — модель не загружена (нет артефактов).
- **`HTTP_ERROR`** — прочие `HTTPException`.
- **`INTERNAL_ERROR`** (500) — необработанное исключение (в ответе без утечки внутренних деталей).

## Endpoints

### `GET /`

Индекс сервиса: версии, базовый путь API, ссылки на документацию.

### `GET /v1/health`

Проверка живости и наличия загруженной модели.

**Пример ответа:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "api_route_version": "v1",
  "service_semantic_version": "1.0.0"
}
```

### `GET /v1/model-info`

Метаданные загруженной модели и версий: размерность входа, порог, архитектура MLP из `training_config.json`, `trained_at_utc` (если сохранён при обучении), `model_weights_modified_at_utc` (mtime файла весов), имена файлов артефактов. Если модель не загружена, `model_loaded: false`, остальные поля модели — `null`.

### `POST /v1/predict`

Одна поездка (признаки в JSON). Схема — `TripFeaturesRequest`.

**Примеры payload:**

- Файл: `examples/single_predict_request.json`
- В Swagger UI (**`/docs`**) — встроенный пример «high_risk_night_trip» и **Example** из схемы (см. `app/core/payload_examples.py`)

**Пример ответа:**

```json
{
  "predicted_class": 1,
  "probability_positive": 0.73,
  "threshold_used": 0.5
}
```

### `POST /v1/predict_batch`

Массив объектов в поле `items` (см. `examples/batch_predict_request.json`; в `/docs` — пример «two_trips»).

**Пример ответа:**

```json
{
  "predictions": [
    {"predicted_class": 0, "probability_positive": 0.12, "threshold_used": 0.5},
    {"predicted_class": 1, "probability_positive": 0.81, "threshold_used": 0.5}
  ]
}
```

## Документация OpenAPI

После запуска: **http://127.0.0.1:8000/docs** (Swagger), **http://127.0.0.1:8000/redoc**.

## Требования

- В `artifacts/` — модель, препроцессор и `training_config.json` (после `scripts/train.py`).
- Переменные окружения: `.env.example`.

## cURL

```bash
curl -s http://127.0.0.1:8000/

curl -s http://127.0.0.1:8000/v1/health

curl -s http://127.0.0.1:8000/v1/model-info

curl -s -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d @examples/single_predict_request.json
```

С явным идентификатором запроса:

```bash
curl -s http://127.0.0.1:8000/v1/health -H "X-Request-ID: my-trace-id"
```
