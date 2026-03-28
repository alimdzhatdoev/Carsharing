"""Операции с датасетами: демо-генерация, список файлов, загрузка, превью, валидация."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.data.loader import load_raw_csv
from app.data.pipeline import profile_dataframe
from app.data.schema import validate_training_dataframe
from app.features.build_features import TARGET_COLUMN
from app.utils.common import get_project_root


def raw_data_dir(root: Path | None = None) -> Path:
    base = root or get_project_root()
    d = base / "data" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_available_datasets(root: Path | None = None) -> list[str]:
    """Имена *.csv в data/raw/."""
    d = raw_data_dir(root)
    return sorted(p.name for p in d.glob("*.csv"))


def generate_demo_dataset(
    rows: int = 5000,
    *,
    filename: str = "trips_demo.csv",
    seed: int = 42,
    root: Path | None = None,
) -> Path:
    from scripts.generate_demo_data import generate_dataset

    out_dir = raw_data_dir(root)
    path = out_dir / filename
    df = generate_dataset(rows, random_seed=seed)
    df.to_csv(path, index=False)
    return path


def save_uploaded_csv(content: bytes, filename: str, root: Path | None = None) -> Path:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"
    path = raw_data_dir(root) / safe_name
    path.write_bytes(content)
    return path


def load_dataset_preview(path: Path | str, n_rows: int = 100) -> pd.DataFrame:
    df = load_raw_csv(path)
    return df.head(n_rows)


def dataset_full(path: Path | str) -> pd.DataFrame:
    return load_raw_csv(path)


def validate_training_csv(path: Path | str, *, strict_categories: bool = False) -> tuple[bool, list[str], list[str]]:
    df = load_raw_csv(path)
    vr = validate_training_dataframe(df, strict_categories=strict_categories, min_rows=20)
    return vr.ok, vr.errors, vr.warnings


def summarize_dataset(path: Path | str) -> dict:
    df = load_raw_csv(path)
    prof = profile_dataframe(df, TARGET_COLUMN)
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_rate": prof["missing_rate_by_column"],
        "target_counts": prof.get("target_value_counts"),
    }
