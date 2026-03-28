"""Train / validation / test splits without leakage."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    First hold out test_size; then split remainder into train/val with val_size
    as fraction of remainder.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[target_column],
    )
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=random_seed,
        stratify=train_val[target_column],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
