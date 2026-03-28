"""
Готовые сценарии и условные веса влияния признаков для демонстрации прогнозирования.

Веса (1–5) — экспертные оценки для защиты, НЕ градиенты обученной модели.
См. пояснение в docs/defense_narrative.md, раздел 7.
"""

from __future__ import annotations

from app.core.payload_examples import TRIP_FEATURES_EXAMPLE
from app.features.build_features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Условная сила влияния на итоговый «риск» в демо (для подписей к полям и справки)
FEATURE_INFLUENCE_1_TO_5: dict[str, int] = {
    "previous_incidents_count": 5,
    "payment_delay_count": 5,
    "city_zone_risk": 5,
    "late_finish_flag": 4,
    "night_trip_flag": 4,
    "max_speed_kmh": 4,
    "average_speed_kmh": 4,
    "traffic_level": 4,
    "weather_condition": 3,
    "rating": 3,
    "trip_start_zone": 2,
    "trip_end_zone": 2,
    "tariff_type": 2,
    "car_category": 2,
    "user_age": 2,
    "driving_experience_years": 3,
    "account_age_days": 2,
    "trip_duration_min": 2,
    "trip_distance_km": 2,
    "trip_cost": 2,
    "start_hour": 3,
    "day_of_week": 2,
    "car_age_years": 2,
    "mileage_km": 2,
    "fuel_or_charge_level_start": 2,
    "fuel_or_charge_level_end": 2,
    "weekend_flag": 2,
}


def influence_stars(key: str) -> str:
    n = max(1, min(5, FEATURE_INFLUENCE_1_TO_5.get(key, 2)))
    return "★" * n + "☆" * (5 - n)


def influence_label(key: str) -> str:
    return f"влияние: {influence_stars(key)}"


def merge_trip_features(overrides: dict) -> dict:
    out = {**TRIP_FEATURES_EXAMPLE}
    out.update(overrides)
    return out


# Готовые сценарии: полное описание + только отличия от TRIP_FEATURES_EXAMPLE
DEMO_SCENARIOS: list[dict] = [
    {
        "key": "baseline_json",
        "title": "Базовый пример (как в демо-JSON)",
        "short": "Ночь, выходной, дождь, высокий трафик и зональный риск — «насыщенный» вектор.",
        "rationale_ru": (
            "Эталонный демо-вектор: поездка ночью в выходной, под дождём, при высокой загруженности дорог, "
            "в зоне с повышенным заявленным риском, с опозданием завершения аренды и уже имевшимися задержками платежей. "
            "В предметной области такой набор факторов обычно заставляет оператора отнестись к поездке внимательнее."
        ),
        "overrides": {},
    },
    {
        "key": "calm_day",
        "title": "Спокойная дневная поездка",
        "short": "Будни, день, хорошая погода, низкий трафик и зона низкого риска, без просрочек.",
        "rationale_ru": (
            "Типичная «спокойная» картина: день в будни, хорошая видимость, низкий трафик и низкий зональный риск, "
            "аренда завершена вовремя, нет задержек по платежам и прошлых инцидентов, скорости умеренные, рейтинг пользователя высокий. "
            "С точки зрения здравого смысла это ближе к безопасному и предсказуемому сценарию."
        ),
        "overrides": {
            "night_trip_flag": 0,
            "weekend_flag": 0,
            "start_hour": 11,
            "day_of_week": 2,
            "traffic_level": "low",
            "city_zone_risk": "low",
            "weather_condition": "clear",
            "late_finish_flag": 0,
            "payment_delay_count": 0,
            "previous_incidents_count": 0,
            "rating": 4.6,
            "max_speed_kmh": 72.0,
            "average_speed_kmh": 32.0,
            "trip_duration_min": 25.0,
            "trip_distance_km": 8.0,
        },
    },
    {
        "key": "speed_risk",
        "title": "Высокая скорость и просрочка",
        "short": "Сохраняем ночь и сложный контекст, усиливаем скорость и флаг позднего завершения.",
        "rationale_ru": (
            "На фоне уже «напряжённого» базового контекста (ночь, погода, трафик) добавлены очень высокие средняя и максимальная скорость, "
            "сохраняется просрочка завершения и учтён хотя бы один прошлый инцидент. Такой профиль часто связывают с агрессивной ездой и повышенной вероятностью инцидента."
        ),
        "overrides": {
            "max_speed_kmh": 168.0,
            "average_speed_kmh": 95.0,
            "late_finish_flag": 1,
            "previous_incidents_count": 1,
        },
    },
    {
        "key": "bad_profile",
        "title": "Проблемный профиль пользователя",
        "short": "Много задержек платежей и инцидентов, низкий рейтинг, премиум и зона высокого риска.",
        "rationale_ru": (
            "Акцент на истории пользователя: многочисленные задержки платежей и инциденты в прошлом, низкий рейтинг, "
            "сложные условия (туман, высокий трафик, высокий зональный риск). Даже при тарифе premium такой профиль "
            "для оператора выглядит как повышенный операционный и мошеннический риск."
        ),
        "overrides": {
            "payment_delay_count": 6,
            "previous_incidents_count": 4,
            "rating": 2.1,
            "tariff_type": "premium",
            "city_zone_risk": "high",
            "traffic_level": "high",
            "weather_condition": "fog",
        },
    },
    {
        "key": "balanced_standard",
        "title": "Обычная стандарт-поездка",
        "short": "Типичный день, средние скорости и тариф standard без крайностей.",
        "rationale_ru": (
            "Обычный будний день, тариф standard, средние скорости и умеренный трафик, зональный риск и погода без крайностей, "
            "нет просрочек и «красных флагов» в платежах и инцидентах, рейтинг нормальный. Это близко к типичной поездке массового пользователя."
        ),
        "overrides": {
            "night_trip_flag": 0,
            "weekend_flag": 0,
            "start_hour": 16,
            "tariff_type": "standard",
            "traffic_level": "medium",
            "city_zone_risk": "medium",
            "weather_condition": "clear",
            "late_finish_flag": 0,
            "payment_delay_count": 0,
            "previous_incidents_count": 0,
            "rating": 4.0,
            "max_speed_kmh": 85.0,
            "average_speed_kmh": 38.0,
        },
    },
    {
        "key": "snow_evening",
        "title": "Снег, вечер, средний трафик",
        "short": "Зимние условия и сумерки без крайностей по скорости и истории пользователя.",
        "rationale_ru": (
            "Погода осложнена (снег), час ближе к вечеру, трафик средний, зональный риск умеренный. "
            "Платежи и инциденты в норме, скорости не завышены — типичный зимний городской сценарий."
        ),
        "overrides": {
            "weather_condition": "snow",
            "start_hour": 19,
            "traffic_level": "medium",
            "city_zone_risk": "medium",
            "night_trip_flag": 0,
            "max_speed_kmh": 55.0,
            "average_speed_kmh": 28.0,
        },
    },
    {
        "key": "rookie_user",
        "title": "Молодой водитель, короткий стаж",
        "short": "Низкий возраст и мало лет за рулём при остальных умеренных признаках.",
        "rationale_ru": (
            "Пользователь молодой, стаж небольшой — в страховании и у операторов это часто отдельный фактор внимания, "
            "даже если поездка короткая и без явных нарушений в данных."
        ),
        "overrides": {
            "user_age": 19,
            "driving_experience_years": 0.5,
            "trip_duration_min": 18.0,
            "trip_distance_km": 5.0,
            "max_speed_kmh": 60.0,
            "average_speed_kmh": 28.0,
            "night_trip_flag": 0,
            "traffic_level": "low",
        },
    },
    {
        "key": "economy_short",
        "title": "Короткая economy по центру",
        "short": "Недорогой тариф, короткая дистанция, день, ясно, низкий риск зоны.",
        "rationale_ru": (
            "Минимальная по длительности и стоимости поездка на economy в центре при хорошей погоде и низкой загрузке дорог. "
            "Профиль обычно ассоциируется с низкой операционной нагрузкой."
        ),
        "overrides": {
            "tariff_type": "economy",
            "trip_start_zone": "center",
            "trip_end_zone": "center",
            "trip_duration_min": 12.0,
            "trip_distance_km": 3.5,
            "trip_cost": 95.0,
            "weather_condition": "clear",
            "traffic_level": "low",
            "city_zone_risk": "low",
            "night_trip_flag": 0,
            "weekend_flag": 0,
            "start_hour": 13,
            "late_finish_flag": 0,
            "payment_delay_count": 0,
            "previous_incidents_count": 0,
            "rating": 4.5,
        },
    },
    {
        "key": "van_high_mileage",
        "title": "Фургон, высокий пробег, ночь",
        "short": "Категория van, большой пробег ТС, ночная поездка при дожде.",
        "rationale_ru": (
            "Крупнее класс автомобиля, высокий пробег парка, ночь и дождь — факторы, которые могут коррелировать с износом ТС "
            "и сложностью управления; для модели это отдельные признаки в векторе."
        ),
        "overrides": {
            "car_category": "van",
            "mileage_km": 185000.0,
            "car_age_years": 9.0,
            "night_trip_flag": 1,
            "weather_condition": "rain",
            "traffic_level": "medium",
            "max_speed_kmh": 88.0,
            "average_speed_kmh": 42.0,
        },
    },
    {
        "key": "stacked_red_flags",
        "title": "Совокупность красных флагов",
        "short": "Инциденты, задержки платежей, низкий рейтинг, ночь, просрочка, высокая скорость и зона риска.",
        "rationale_ru": (
            "Намеренно перегруженный негативом сценарий: история пользователя плохая, условия поездки тяжёлые, "
            "скорости высокие, завершение с просрочкой. На защите полезно показать, как модель реагирует на «худший» состав признаков."
        ),
        "overrides": {
            "payment_delay_count": 8,
            "previous_incidents_count": 5,
            "rating": 1.4,
            "city_zone_risk": "high",
            "traffic_level": "high",
            "weather_condition": "rain",
            "night_trip_flag": 1,
            "late_finish_flag": 1,
            "max_speed_kmh": 175.0,
            "average_speed_kmh": 102.0,
            "trip_cost": 890.0,
            "trip_duration_min": 95.0,
        },
    },
]


def scenario_feature_dict(scenario: dict) -> dict:
    return merge_trip_features(scenario.get("overrides") or {})


def diff_vs_baseline(overrides: dict, baseline: dict | None = None) -> dict:
    base = baseline or TRIP_FEATURES_EXAMPLE
    changes = {}
    merged = merge_trip_features(overrides)
    for k in set(base) | set(merged):
        if k not in merged:
            continue
        bv, mv = base.get(k), merged.get(k)
        if bv != mv:
            changes[k] = (bv, mv)
    return changes


def all_feature_keys_ordered() -> list[str]:
    """Алфавитный порядок: числовые, затем категориальные (для справочных таблиц)."""
    return list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)


def feature_keys_by_influence_desc() -> list[str]:
    """Все признаки модели по убыванию условного веса (для формы и колонок таблицы)."""
    keys = list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)
    return sorted(
        keys,
        key=lambda k: (-FEATURE_INFLUENCE_1_TO_5.get(k, 2), k),
    )
