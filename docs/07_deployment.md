# Развёртывание

## Локально

1. Установка зависимостей: `pip install -r requirements.txt`
2. Генерация данных: `python scripts/generate_demo_data.py`
3. Обучение: `python scripts/train.py`
4. API: `uvicorn app.api.main:app --host 0.0.0.0 --port 8000`

## Docker

```bash
docker compose build
docker compose up -d
```

Проверка: `GET http://localhost:8000/v1/health`, `GET http://localhost:8000/v1/model-info`

Образ ожидает, что артефакты **смонтированы** в контейнер (volume `./artifacts`) или собраны в образ на этапе CI. В `docker-compose.yml` по умолчанию монтируются `artifacts` из хоста после локального обучения.

## Переменные окружения

См. `.env.example`: пути к артефактам, порог, хост/порт при необходимости.

## Прод-рекомендации (вне демо)

- Отдельный registry образов, секреты через vault, не коммитить веса в git при больших размерах.
- Мониторинг: latency, ошибки, дрейф распределений входов.
- Версионирование модели и препроцессора (MLflow или простые теги в имени файла).
- A/B тестирование порога и модели.
