import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.core.config import ModelConfig
from app.data.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    save_preprocessor,
    split_features_target,
    transform_features,
)
from app.models.net import TabularMLP
from app.models.predict import load_model_for_inference, predict_one
from scripts.generate_demo_data import generate_dataset


def test_predict_one_roundtrip():
    df = generate_dataset(80, random_seed=3)
    train = df.iloc[:60]
    X_train, _ = split_features_target(train)
    pre = build_preprocessor()
    fit_preprocessor(pre, X_train)
    Xt = transform_features(pre, X_train)
    input_dim = Xt.shape[1]

    cfg = ModelConfig(hidden_layers=[16], dropout=0.0, use_batch_norm=False, activation="relu")
    model = TabularMLP(input_dim, cfg)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.pt"
        torch.save({"state_dict": model.state_dict()}, p)
        meta = {"input_dim": input_dim, "model": cfg.model_dump()}
        loaded = load_model_for_inference(p, meta, cfg, device=torch.device("cpu"))

    row = X_train.iloc[0].to_dict()
    out = predict_one(loaded, pre, row, torch.device("cpu"), 0.5)
    assert "predicted_class" in out
    assert "probability_positive" in out
    assert out["predicted_class"] in (0, 1)
    assert 0 <= out["probability_positive"] <= 1


def test_save_load_preprocessor_consistency():
    df = generate_dataset(50, random_seed=4)
    X, _ = split_features_target(df.iloc[:40])
    pre = build_preprocessor()
    fit_preprocessor(pre, X)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pre.joblib"
        save_preprocessor(pre, path)
        from app.data.preprocessing import load_preprocessor

        pre2 = load_preprocessor(path)
    df_new = generate_dataset(10, random_seed=5)
    X2, _ = split_features_target(df_new)
    a = transform_features(pre, X2)
    b = transform_features(pre2, X2)
    np.testing.assert_allclose(a, b)
