"""Фасад предсказаний для UI (обёртка над InferenceService)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import AppConfig, EnvOverrides, load_app_config, merge_env_overrides, resolved_paths
from app.services.inference_service import InferenceService
from app.utils.common import get_project_root


def build_inference_service(cfg: AppConfig | None = None) -> InferenceService:
    root = get_project_root()
    if cfg is None:
        cfg = merge_env_overrides(load_app_config(root=root), EnvOverrides())
    cfg = resolved_paths(cfg, root)
    svc = InferenceService(cfg)
    svc.load()
    return svc


def predict_single(svc: InferenceService, features: dict[str, Any]) -> dict[str, Any]:
    return svc.predict_one(features)


def predict_batch_dataframe(svc: InferenceService, df: pd.DataFrame) -> pd.DataFrame:
    return svc.predict_from_feature_dataframe(df)
