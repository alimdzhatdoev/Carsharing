"""Metric computation and reporting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from app.data.dataset import TabularDataset
from app.models.net import TabularMLP


def _dataset_stats_and_curves(
    y_true_i: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Размер теста, баланс классов, кривые ROC и PR (для визуализации в UI)."""
    n = int(len(y_true_i))
    n_pos = int(np.sum(y_true_i == 1))
    n_neg = int(np.sum(y_true_i == 0))
    extra: dict[str, Any] = {
        "n_test": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence_positive": float(y_true_i.mean()) if n else 0.0,
        "threshold_used": float(threshold),
    }
    if n < 2 or len(np.unique(y_true_i)) < 2:
        extra["roc_curve"] = None
        extra["pr_curve"] = None
        return extra
    fpr, tpr, roc_thr = roc_curve(y_true_i, proba)
    prec, rec, pr_thr = precision_recall_curve(y_true_i, proba)
    extra["roc_curve"] = {
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
        "thresholds": [float(x) for x in roc_thr],
    }
    extra["pr_curve"] = {
        "precision": [float(x) for x in prec],
        "recall": [float(x) for x in rec],
        "thresholds": [float(x) for x in pr_thr],
    }
    return extra


@torch.no_grad()
def predict_logits(model: TabularMLP, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).squeeze(-1)
        logits.append(out.cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def binary_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    proba = 1.0 / (1.0 + np.exp(-logits))
    y_pred = (proba >= threshold).astype(np.int64)
    y_true_i = y_true.astype(np.int64)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "precision": float(precision_score(y_true_i, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_i, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_i, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true_i)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true_i, proba))
    else:
        out["roc_auc"] = None
    out["confusion_matrix"] = confusion_matrix(y_true_i, y_pred).tolist()
    out["classification_report"] = classification_report(
        y_true_i, y_pred, digits=4, zero_division=0
    )
    out.update(_dataset_stats_and_curves(y_true_i, proba, threshold))
    return out


def binary_metrics_from_proba(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Метрики по уже посчитанной вероятности положительного класса (sklearn baselines)."""
    y_true_i = y_true.astype(np.int64)
    y_pred = (proba >= threshold).astype(np.int64)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_i, y_pred)),
        "precision": float(precision_score(y_true_i, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_i, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_i, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true_i)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true_i, proba))
    else:
        out["roc_auc"] = None
    out["confusion_matrix"] = confusion_matrix(y_true_i, y_pred).tolist()
    out.update(_dataset_stats_and_curves(y_true_i, proba, threshold))
    return out


def save_confusion_matrix_plot(
    cm: np.ndarray,
    path: Path,
    title: str = "Матрица ошибок (тестовая выборка)",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Прогноз: 0", "Прогноз: 1"],
        yticklabels=["Истина: 0", "Истина: 1"],
        title=title,
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_model(
    model: TabularMLP,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    logits, y = predict_logits(model, loader, device)
    return binary_metrics(y, logits, threshold)
