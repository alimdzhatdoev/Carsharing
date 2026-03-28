"""Batch and single inference from raw feature rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from app.core.config import ModelConfig
from app.data.preprocessing import load_preprocessor, transform_features
from app.features.build_features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from app.models.net import TabularMLP
from app.models.utils import get_device


def features_dataframe(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    cols = list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)
    df = pd.DataFrame(list(rows))
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    return df[cols]


@torch.no_grad()
def predict_batch(
    model: TabularMLP,
    preprocessor,
    rows: Sequence[dict[str, Any]],
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    df = features_dataframe(rows)
    X = transform_features(preprocessor, df)
    xb = torch.from_numpy(X).to(device)
    model.eval()
    logits = model(xb).squeeze(-1)
    proba = torch.sigmoid(logits).cpu().numpy()
    out = []
    for p in proba:
        cls = int(p >= threshold)
        out.append(
            {
                "predicted_class": cls,
                "probability_positive": float(p),
                "threshold_used": threshold,
            }
        )
    return out


def predict_one(
    model: TabularMLP,
    preprocessor,
    row: dict[str, Any],
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    return predict_batch(model, preprocessor, [row], device, threshold)[0]


def load_model_for_inference(
    model_path: Path,
    meta: dict[str, Any],
    model_cfg: ModelConfig,
    device: torch.device | None = None,
) -> TabularMLP:
    device = device or get_device()
    try:
        payload = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location=device)
    input_dim = int(meta["input_dim"])
    model = TabularMLP(input_dim, model_cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_training_meta(path: Path) -> dict[str, Any]:
    import json

    with path.open(encoding="utf-8") as f:
        return json.load(f)
