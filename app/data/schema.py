"""
Контракт сырого датасета для обучения: колонки, базовые проверки, опциональный strict по категориям.

Решение по дизайну: списки признаков не дублируются — импорт из build_features.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.features.build_features import (
    CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

# Демо-справочники; на проде расширять или отключать strict-проверку.
CATEGORICAL_ALLOWED: dict[str, frozenset[str]] = {
    "tariff_type": frozenset({"economy", "standard", "premium"}),
    "trip_start_zone": frozenset({"north", "south", "east", "west", "center"}),
    "trip_end_zone": frozenset({"north", "south", "east", "west", "center"}),
    "car_category": frozenset({"compact", "sedan", "suv", "van"}),
    "weather_condition": frozenset({"clear", "rain", "snow", "fog"}),
    "traffic_level": frozenset({"low", "medium", "high"}),
    "city_zone_risk": frozenset({"low", "medium", "high"}),
}


def expected_training_columns() -> tuple[str, ...]:
    return tuple(IDENTIFIER_COLUMNS) + tuple(NUMERIC_COLUMNS) + tuple(CATEGORICAL_COLUMNS) + (TARGET_COLUMN,)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    target_positive_rate: float | None = None


def validate_training_dataframe(
    df: pd.DataFrame,
    *,
    strict_categories: bool = False,
    min_rows: int = 50,
) -> ValidationResult:
    """
    Проверка кадра перед сплитом и обучением.

    strict_categories: значения категорий должны входить в CATEGORICAL_ALLOWED (демо).
    """
    errors: list[str] = []
    warnings: list[str] = []
    expected = expected_training_columns()

    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        errors.append(f"Duplicate column names: {dup[:10]}")

    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in expected]
    if extra:
        warnings.append(f"Extra columns will be ignored by schema (remainder drop in preprocessor): {extra[:15]}")

    row_count = len(df)
    if row_count < min_rows:
        errors.append(f"Too few rows for stable training: {row_count} < {min_rows}")

    pos_rate: float | None = None
    if TARGET_COLUMN in df.columns and row_count > 0:
        t = df[TARGET_COLUMN]
        if t.isna().any():
            errors.append(f"target_column '{TARGET_COLUMN}' contains NaN")
        elif not t.isin([0, 1]).all():
            errors.append("target_column must contain only 0 and 1")
        else:
            pos_rate = float((t == 1).mean())
            if pos_rate == 0.0 or pos_rate == 1.0:
                warnings.append("Target is single-class; metrics and stratify may degrade")

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col]
        if col in (
            "late_finish_flag",
            "night_trip_flag",
            "weekend_flag",
            "start_hour",
            "day_of_week",
            "payment_delay_count",
            "previous_incidents_count",
        ):
            continue
        if col in ("user_age",):
            bad = s.dropna()[(s.dropna() < 0) | (s.dropna() > 120)]
            if len(bad):
                warnings.append(f"Column '{col}' has values outside plausible range [0,120]")

    if strict_categories:
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            allowed = CATEGORICAL_ALLOWED.get(col)
            if not allowed:
                continue
            vals = df[col].dropna().astype(str).unique()
            unknown = sorted(set(vals) - set(allowed))
            if unknown:
                errors.append(f"Column '{col}' has unknown categories (strict): {unknown[:20]}")

    ok = len(errors) == 0
    return ValidationResult(
        ok=ok,
        errors=errors,
        warnings=warnings,
        row_count=row_count,
        target_positive_rate=pos_rate,
    )
