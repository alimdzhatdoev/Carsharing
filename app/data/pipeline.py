"""
Оркестрация этапов data layer: загрузка, валидация, профилирование.

Не выполняет train/split — это ответственность scripts/train.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.loader import load_raw_csv
from app.data.schema import ValidationResult, validate_training_dataframe


def profile_dataframe(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Лёгкий профиль для аудита и воспроизводимости (без PII в отдельных полях)."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    miss = {c: float(df[c].isna().mean()) for c in df.columns}
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "missing_rate_by_column": miss,
        "numeric_summary": {},
    }
    if target_column in df.columns:
        vc = df[target_column].value_counts(dropna=False).to_dict()
        out["target_value_counts"] = {str(k): int(v) for k, v in vc.items()}
    for c in numeric_cols[:30]:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        out["numeric_summary"][c] = {
            "mean": float(s.mean()),
            "std": float(s.std()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return out


def write_profile(profile: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def load_and_validate_training_csv(
    path: Path | str,
    *,
    strict_categories: bool = False,
    min_rows: int = 50,
) -> tuple[pd.DataFrame, ValidationResult]:
    df = load_raw_csv(path)
    result = validate_training_dataframe(df, strict_categories=strict_categories, min_rows=min_rows)
    return df, result


def run_prepare_stage(
    input_path: Path | str,
    *,
    strict_categories: bool = False,
    profile_path: Path | None = None,
    target_column: str = "target_class",
) -> tuple[pd.DataFrame, ValidationResult]:
    """
    Полный проход для CLI prepare_data: load → validate → optional profile.
    """
    df, vr = load_and_validate_training_csv(
        input_path, strict_categories=strict_categories, min_rows=50
    )
    if profile_path is not None and vr.ok:
        prof = profile_dataframe(df, target_column=target_column)
        write_profile(prof, profile_path)
    return df, vr
