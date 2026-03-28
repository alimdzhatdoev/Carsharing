"""Training loop, early stopping, artifact export."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.core.config import AppConfig
from app.data.dataset import TabularDataset
from app.models.evaluate import evaluate_model, save_confusion_matrix_plot
from app.models.net import TabularMLP
from app.models.utils import get_device


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_roc_auc: float | None


def _run_epoch(
    model: TabularMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        n += len(yb)
    return total_loss / max(n, 1)


@torch.no_grad()
def _val_loss(
    model: TabularMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb).squeeze(-1)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(yb)
        n += len(yb)
    return total_loss / max(n, 1)


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Positive class weight for BCEWithLogitsLoss (handle edge cases)."""
    y = y_train.astype(np.int64)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    w = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    return w


def train_tabular_classifier(
    cfg: AppConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    device: torch.device | None = None,
    on_epoch_end: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[TabularMLP, list[dict[str, Any]], dict[str, Any]]:
    device = device or get_device()
    model = TabularMLP(input_dim, cfg.model).to(device)
    pos_w = compute_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    train_loader = DataLoader(
        TabularDataset(X_train, y_train),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    patience_left = cfg.training.early_stopping_patience

    for epoch in range(1, cfg.training.epochs + 1):
        tr_loss = _run_epoch(model, train_loader, optimizer, criterion, device)
        va_loss = _val_loss(model, val_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, device, cfg.inference.classification_threshold)
        row = EpochLog(
            epoch=epoch,
            train_loss=tr_loss,
            val_loss=va_loss,
            val_accuracy=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_roc_auc=val_metrics.get("roc_auc"),
        )
        row_dict = asdict(row)
        history.append(row_dict)
        if on_epoch_end is not None:
            on_epoch_end(row_dict)

        improved = va_loss < (best_val_loss - cfg.training.early_stopping_min_delta)
        if improved:
            best_val_loss = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = cfg.training.early_stopping_patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "pos_weight": float(pos_w.item()),
    }
    return model, history, summary


def save_model_checkpoint(
    model: TabularMLP,
    path: Path,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(payload, path)


def save_training_history(history: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_classification_artifacts(
    model: TabularMLP,
    cfg: AppConfig,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    paths: dict[str, Path],
) -> dict[str, Any]:
    test_loader = DataLoader(
        TabularDataset(X_test, y_test),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    metrics = evaluate_model(model, test_loader, device, cfg.inference.classification_threshold)
    report_path = paths["reports_dir"] / "classification_report_test.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(metrics["classification_report"], encoding="utf-8")

    cm = np.array(metrics["confusion_matrix"])
    save_confusion_matrix_plot(cm, paths["confusion_matrix_path"])

    metrics_path = paths["metrics_dir"] / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return metrics
