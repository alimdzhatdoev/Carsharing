"""Load raw tabular data."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_csv(path: Path | str, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Read training/inference CSV. Column contract for training: `app/data/schema.py`
    and `docs/03_data_description.md`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    return pd.read_csv(p, **read_csv_kwargs)
