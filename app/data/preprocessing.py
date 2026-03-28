"""Sklearn preprocessing: fit on train only, same transform at inference."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.features.build_features import (
    CATEGORICAL_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    y = df[TARGET_COLUMN].values.astype(np.float32)
    drop_cols = list(IDENTIFIER_COLUMNS) + [TARGET_COLUMN]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def _numeric_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_preprocessor_for_columns(
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> ColumnTransformer:
    """Препроцессор на заданных подмножествах колонок (для ablation / экспериментов)."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", _numeric_pipeline(), list(numeric_cols)))
    if categorical_cols:
        transformers.append(("cat", _categorical_pipeline(), list(categorical_cols)))
    if not transformers:
        raise ValueError("At least one of numeric_cols or categorical_cols must be non-empty")
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor() -> ColumnTransformer:
    return build_preprocessor_for_columns(NUMERIC_COLUMNS, CATEGORICAL_COLUMNS)


def fit_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    preprocessor.fit(X_train)
    return preprocessor


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    out = preprocessor.transform(X)
    return np.asarray(out, dtype=np.float32)


def save_preprocessor(preprocessor: ColumnTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    return joblib.load(path)


def output_feature_dim(preprocessor: ColumnTransformer, X_sample: pd.DataFrame) -> int:
    return int(transform_features(preprocessor, X_sample).shape[1])
