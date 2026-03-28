import numpy as np
import pandas as pd

from app.data.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    split_features_target,
    transform_features,
)
from app.features.build_features import TARGET_COLUMN
from scripts.generate_demo_data import generate_dataset


def test_preprocessor_fit_transform_shape():
    df = generate_dataset(200, random_seed=1)
    train = df.iloc[:120]
    val = df.iloc[120:160]

    X_train, y_train = split_features_target(train)
    X_val, _ = split_features_target(val)

    pre = build_preprocessor()
    fit_preprocessor(pre, X_train)
    Xt_train = transform_features(pre, X_train)
    Xt_val = transform_features(pre, X_val)

    assert Xt_train.shape[0] == len(X_train)
    assert Xt_val.shape[0] == len(X_val)
    assert Xt_train.shape[1] == Xt_val.shape[1]
    assert not np.isnan(Xt_train).any()
    assert y_train.shape[0] == len(train)
    assert set(np.unique(y_train)) <= {0.0, 1.0}


def test_no_target_in_features():
    df = generate_dataset(30, random_seed=2)
    X, _ = split_features_target(df)
    assert TARGET_COLUMN not in X.columns
    assert "user_id" not in X.columns
