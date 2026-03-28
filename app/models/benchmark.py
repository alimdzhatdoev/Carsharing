"""
Сводные эксперименты: ablations × baseline-модели (+ опционально короткий MLP на каждом срезе).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.core.config import AppConfig
from app.data.dataset import TabularDataset
from app.data.preprocessing import (
    build_preprocessor_for_columns,
    fit_preprocessor,
    transform_features,
)
from app.features.ablation import ABLATION_DROP, ablation_feature_columns, subset_features
from app.models.baselines import fit_predict_baselines
from app.models.evaluate import evaluate_model
from app.models.train import train_tabular_classifier


def run_ablation_benchmark(
    cfg: AppConfig,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    ablation_names: list[str] | None = None,
    include_xgboost: bool = True,
    train_mlp_per_ablation: bool = False,
    mlp_epochs_cap: int = 35,
    mlp_patience: int = 5,
) -> list[dict[str, Any]]:
    """
    Для каждого ablation: отдельный fit препроцессора на train, baselines на матрице признаков,
    опционально — укороченное обучение MLP (те же веса классов через pos_weight внутри train).
    """
    names = ablation_names or list(ABLATION_DROP.keys())
    rows: list[dict[str, Any]] = []
    threshold = cfg.inference.classification_threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ab in names:
        dropped = ABLATION_DROP.get(ab, frozenset())
        num_cols, cat_cols = ablation_feature_columns(dropped)
        Xtr = subset_features(X_train, dropped)
        Xva = subset_features(X_val, dropped)
        Xte = subset_features(X_test, dropped)

        pre = build_preprocessor_for_columns(num_cols, cat_cols)
        fit_preprocessor(pre, Xtr)
        Xt_tr = transform_features(pre, Xtr)
        Xt_va = transform_features(pre, Xva)
        Xt_te = transform_features(pre, Xte)

        for br in fit_predict_baselines(
            Xt_tr,
            y_train,
            Xt_te,
            y_test,
            threshold=threshold,
            random_state=cfg.data.random_seed,
            include_xgboost=include_xgboost,
        ):
            note = br.metrics.get("note")
            row = {
                "ablation": ab,
                "model": br.name,
                "accuracy": br.metrics.get("accuracy"),
                "precision": br.metrics.get("precision"),
                "recall": br.metrics.get("recall"),
                "f1": br.metrics.get("f1"),
                "roc_auc": br.metrics.get("roc_auc"),
                "dropped_columns": sorted(dropped),
                "input_dim": int(Xt_tr.shape[1]),
            }
            if note:
                row["note"] = note
            rows.append(row)

        if train_mlp_per_ablation:
            cfg_m = cfg.model_copy(deep=True)
            cfg_m.training.epochs = min(mlp_epochs_cap, cfg_m.training.epochs)
            cfg_m.training.early_stopping_patience = mlp_patience
            model, _, summ = train_tabular_classifier(
                cfg_m,
                Xt_tr,
                y_train,
                Xt_va,
                y_val,
                input_dim=Xt_tr.shape[1],
                device=device,
            )
            loader = DataLoader(
                TabularDataset(Xt_te, y_test),
                batch_size=cfg_m.training.batch_size,
                shuffle=False,
                num_workers=cfg_m.training.num_workers,
            )
            m = evaluate_model(model, loader, device, threshold)
            rows.append(
                {
                    "ablation": ab,
                    "model": "mlp_short",
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m.get("roc_auc"),
                    "dropped_columns": sorted(dropped),
                    "input_dim": int(Xt_tr.shape[1]),
                    "mlp_epochs_ran": summ.get("epochs_ran"),
                }
            )

    return rows


def attach_production_mlp_metrics(
    rows: list[dict[str, Any]],
    test_metrics_path: str | Path,
) -> list[dict[str, Any]]:
    """Добавить строку с метриками основного MLP из artifacts (только ablation=full)."""
    p = Path(test_metrics_path)
    if not p.exists():
        return rows
    with p.open(encoding="utf-8") as f:
        m = json.load(f)
    extra = {
        "ablation": "full",
        "model": "mlp_production",
        "accuracy": m.get("accuracy"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
        "roc_auc": m.get("roc_auc"),
        "dropped_columns": [],
        "input_dim": None,
        "note": "from artifacts/metrics/test_metrics.json after scripts/train.py",
    }
    return rows + [extra]


def results_to_markdown_table(rows: list[dict[str, Any]]) -> str:
    cols = ["ablation", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
