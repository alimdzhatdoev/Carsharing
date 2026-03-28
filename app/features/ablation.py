"""
Ablation-спеки: какие признаки убрать из X перед препроцессингом.

Ключ — множество имён колонок для исключения (из NUMERIC_COLUMNS ∪ CATEGORICAL_COLUMNS).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.features.build_features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Имя эксперимента -> колонки, которые **не** подаём в модель для этого прогона.
ABLATION_DROP: dict[str, frozenset[str]] = {
    "full": frozenset(),
    "wo_context": frozenset(
        {
            "night_trip_flag",
            "weekend_flag",
            "weather_condition",
            "traffic_level",
            "city_zone_risk",
        }
    ),
    "wo_user_history": frozenset(
        {
            "payment_delay_count",
            "previous_incidents_count",
            "rating",
        }
    ),
    "wo_trip_dynamics": frozenset(
        {
            "trip_duration_min",
            "trip_distance_km",
            "average_speed_kmh",
            "max_speed_kmh",
            "late_finish_flag",
        }
    ),
}


def ablation_feature_columns(dropped: Iterable[str]) -> tuple[list[str], list[str]]:
    d = set(dropped)
    num = [c for c in NUMERIC_COLUMNS if c not in d]
    cat = [c for c in CATEGORICAL_COLUMNS if c not in d]
    return num, cat


def subset_features(X: pd.DataFrame, dropped: Iterable[str]) -> pd.DataFrame:
    drop = [c for c in dropped if c in X.columns]
    return X.drop(columns=drop, errors="ignore")


def list_ablation_names() -> list[str]:
    return list(ABLATION_DROP.keys())
