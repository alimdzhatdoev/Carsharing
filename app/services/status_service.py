"""Сводный статус проекта и артефактов для Overview / UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import AppConfig, EnvOverrides, load_app_config, merge_env_overrides, resolved_paths
from app.services.evaluation_service import read_test_metrics, read_training_history
from app.utils.common import get_project_root


def get_artifacts_status(cfg: AppConfig) -> dict[str, Any]:
    """Какие ключевые файлы существуют."""
    return {
        "model_pt": Path(cfg.paths.model_path).exists(),
        "preprocessor_joblib": Path(cfg.paths.preprocessor_path).exists(),
        "training_config_json": Path(cfg.paths.training_config_dump).exists(),
        "test_metrics_json": (Path(cfg.paths.metrics_dir) / "test_metrics.json").exists(),
        "training_history_json": Path(cfg.paths.history_path).exists(),
        "test_split_csv": (Path(cfg.paths.processed_dir) / "test_split.csv").exists(),
        "model_report_md": (Path(cfg.paths.reports_dir) / "model_report.md").exists(),
    }


def model_is_trained(cfg: AppConfig) -> bool:
    st = get_artifacts_status(cfg)
    return st["model_pt"] and st["preprocessor_joblib"] and st["training_config_json"]


def load_training_meta_dict(cfg: AppConfig) -> dict[str, Any] | None:
    p = Path(cfg.paths.training_config_dump)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def get_project_overview(cfg: AppConfig | None = None) -> dict[str, Any]:
    root = get_project_root()
    if cfg is None:
        cfg = merge_env_overrides(load_app_config(root=root), EnvOverrides())
    cfg = resolved_paths(cfg, root)

    art = get_artifacts_status(cfg)
    trained = model_is_trained(cfg)
    meta = load_training_meta_dict(cfg) if trained else None
    metrics = read_test_metrics(cfg) if trained else None
    history = read_training_history(cfg) if trained else None

    raw_csv = Path(cfg.paths.raw_data)
    return {
        "config_raw_data_path": cfg.paths.raw_data,
        "raw_dataset_exists": raw_csv.exists(),
        "raw_dataset_path": str(raw_csv) if raw_csv.exists() else None,
        "model_trained": trained,
        "artifacts": art,
        "trained_at_utc": meta.get("trained_at_utc") if meta else None,
        "input_dim": meta.get("input_dim") if meta else None,
        "classification_threshold": meta.get("classification_threshold") if meta else None,
        "test_metrics": metrics,
        "epochs_ran": len(history) if history else None,
    }
