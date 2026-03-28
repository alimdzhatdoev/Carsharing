"""Оркестрация полного цикла обучения (вызывается из UI и scripts/train.py)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from app.core.api_constants import ARTIFACT_META_SCHEMA_VERSION
from app.core.config import AppConfig
from app.core.logger import setup_logging
from app.data.loader import load_raw_csv
from app.data.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    save_preprocessor,
    split_features_target,
    transform_features,
)
from app.data.split import train_val_test_split
from app.models.train import (
    save_classification_artifacts,
    save_model_checkpoint,
    save_training_history,
    train_tabular_classifier,
)
from app.models.utils import set_seed
from app.utils.common import get_project_root


@dataclass
class TrainingResult:
    success: bool
    message: str
    test_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] | None = None
    training_summary: dict[str, Any] | None = None
    log_lines: list[str] = field(default_factory=list)
    model_path: str | None = None


class _ListHandler(logging.Handler):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self._lines = lines

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._lines.append(self.format(record))
        except Exception:
            pass


def run_full_training(
    cfg: AppConfig,
    *,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    capture_logs: bool = True,
) -> TrainingResult:
    """
    Полный пайплайн: сплит, препроцессинг, обучение MLP, сохранение артефактов и тестовых метрик.
    cfg должен быть уже resolved_paths (абсолютные пути).
    """
    log_lines: list[str] = []
    handler: _ListHandler | None = None
    root_logger = logging.getLogger()
    if capture_logs:
        setup_logging()
        handler = _ListHandler(log_lines)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger.addHandler(handler)

    try:
        set_seed(cfg.data.random_seed)
        raw_path = Path(cfg.paths.raw_data)
        if not raw_path.exists():
            return TrainingResult(
                success=False,
                message=f"Файл данных не найден: {raw_path}",
                log_lines=log_lines,
            )

        df = load_raw_csv(raw_path)
        train_df, val_df, test_df = train_val_test_split(
            df,
            cfg.data.target_column,
            cfg.data.test_size,
            cfg.data.val_size,
            cfg.data.random_seed,
        )

        processed_dir = Path(cfg.paths.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(processed_dir / "test_split.csv", index=False)

        X_train, y_train = split_features_target(train_df)
        X_val, y_val = split_features_target(val_df)
        X_test, y_test = split_features_target(test_df)

        preprocessor = build_preprocessor()
        fit_preprocessor(preprocessor, X_train)
        save_preprocessor(preprocessor, Path(cfg.paths.preprocessor_path))

        Xt_train = transform_features(preprocessor, X_train)
        Xt_val = transform_features(preprocessor, X_val)
        Xt_test = transform_features(preprocessor, X_test)

        input_dim = int(Xt_train.shape[1])
        device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, history, summary = train_tabular_classifier(
            cfg,
            Xt_train,
            y_train,
            Xt_val,
            y_val,
            input_dim=input_dim,
            device=device_train,
            on_epoch_end=epoch_callback,
        )

        meta = {
            "input_dim": input_dim,
            "classification_threshold": cfg.inference.classification_threshold,
            "model": cfg.model.model_dump(),
            "target_column": cfg.data.target_column,
            "random_seed": cfg.data.random_seed,
            "artifact_schema_version": ARTIFACT_META_SCHEMA_VERSION,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        model_path = Path(cfg.paths.model_path)
        save_model_checkpoint(model, model_path, meta)

        meta_path = Path(cfg.paths.training_config_dump)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        save_training_history(history, Path(cfg.paths.history_path))

        paths_map = {
            "reports_dir": Path(cfg.paths.reports_dir),
            "metrics_dir": Path(cfg.paths.metrics_dir),
            "confusion_matrix_path": Path(cfg.paths.confusion_matrix_path),
        }
        test_metrics = save_classification_artifacts(
            model,
            cfg,
            Xt_test,
            y_test,
            device_train,
            paths_map,
        )

        return TrainingResult(
            success=True,
            message="Обучение завершено успешно.",
            test_metrics={k: test_metrics[k] for k in test_metrics if k != "classification_report"},
            history=history,
            training_summary=summary,
            log_lines=log_lines,
            model_path=str(model_path),
        )
    except Exception as e:
        logging.getLogger(__name__).exception("training failed")
        return TrainingResult(
            success=False,
            message=str(e),
            log_lines=log_lines,
        )
    finally:
        if handler is not None:
            root_logger.removeHandler(handler)
