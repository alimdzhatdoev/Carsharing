"""Loads artifacts once and serves predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from app.core.api_constants import (
    API_ROUTE_VERSION,
    ARTIFACT_META_SCHEMA_VERSION,
    SERVICE_SEMANTIC_VERSION,
)
from app.core.config import (
    AppConfig,
    EnvOverrides,
    ModelConfig,
    load_app_config,
    merge_env_overrides,
    resolved_paths,
)
from app.core.logger import get_logger
from app.data.preprocessing import load_preprocessor
from app.features.build_features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from app.models.predict import load_model_for_inference, load_training_meta, predict_batch, predict_one
from app.utils.common import get_project_root

logger = get_logger(__name__)


class InferenceService:
    def __init__(self, cfg: AppConfig | None = None):
        root = get_project_root()
        if cfg is None:
            cfg = load_app_config(root=root)
            cfg = merge_env_overrides(cfg, EnvOverrides())
        self.cfg = resolved_paths(cfg, root)
        self._root = root
        self._device = torch.device("cpu")
        self._model = None
        self._preprocessor = None
        self._meta: dict[str, Any] = {}
        self._threshold = self.cfg.inference.classification_threshold
        self._model_path: Path | None = None
        self._meta_path: Path | None = None

    def load(self) -> None:
        model_path = Path(self.cfg.paths.model_path)
        prep_path = Path(self.cfg.paths.preprocessor_path)
        meta_path = Path(self.cfg.paths.training_config_dump)
        self._model_path = model_path
        self._meta_path = meta_path
        if not model_path.exists() or not prep_path.exists() or not meta_path.exists():
            logger.warning(
                "Artifacts missing (model/preprocessor/meta). Predictions will fail until training."
            )
            return
        self._meta = load_training_meta(meta_path)
        model_cfg = ModelConfig.model_validate(self._meta["model"])
        self._preprocessor = load_preprocessor(prep_path)
        self._model = load_model_for_inference(
            model_path,
            self._meta,
            model_cfg,
            device=self._device,
        )
        self._threshold = float(self._meta.get("classification_threshold", self._threshold))
        logger.info("Inference artifacts loaded from %s", model_path.parent)

    @property
    def ready(self) -> bool:
        return self._model is not None and self._preprocessor is not None

    def predict_one(self, row: dict[str, Any]) -> dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Model not loaded. Train the model and ensure artifacts exist.")
        assert self._model is not None and self._preprocessor is not None
        return predict_one(self._model, self._preprocessor, row, self._device, self._threshold)

    def predict_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.ready:
            raise RuntimeError("Model not loaded. Train the model and ensure artifacts exist.")
        assert self._model is not None and self._preprocessor is not None
        return predict_batch(self._model, self._preprocessor, rows, self._device, self._threshold)

    def predict_from_feature_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Строки = объекты; нужны все колонки признаков (числовые + категориальные)."""
        if not self.ready:
            raise RuntimeError("Model not loaded. Train the model and ensure artifacts exist.")
        cols = list(NUMERIC_COLUMNS) + list(CATEGORICAL_COLUMNS)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Не хватает колонок: {sorted(missing)}")
        rows = df[cols].to_dict("records")
        preds = self.predict_batch(rows)
        out = df.copy()
        out["predicted_class"] = [p["predicted_class"] for p in preds]
        out["probability_positive"] = [p["probability_positive"] for p in preds]
        out["threshold_used"] = preds[0]["threshold_used"] if preds else self._threshold
        return out

    def get_model_info(self) -> dict[str, Any]:
        """Метаданные для GET /v1/model-info (без чтения весов)."""
        if not self.ready or self._model_path is None:
            return {
                "model_loaded": False,
                "api_route_version": API_ROUTE_VERSION,
                "service_semantic_version": SERVICE_SEMANTIC_VERSION,
                "artifact_schema_version": ARTIFACT_META_SCHEMA_VERSION,
            }
        mtime = datetime.fromtimestamp(
            self._model_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        return {
            "model_loaded": True,
            "api_route_version": API_ROUTE_VERSION,
            "service_semantic_version": SERVICE_SEMANTIC_VERSION,
            "artifact_schema_version": self._meta.get(
                "artifact_schema_version", ARTIFACT_META_SCHEMA_VERSION
            ),
            "trained_at_utc": self._meta.get("trained_at_utc"),
            "model_weights_modified_at_utc": mtime,
            "input_dim": self._meta.get("input_dim"),
            "classification_threshold": float(
                self._meta.get("classification_threshold", self._threshold)
            ),
            "architecture": self._meta.get("model"),
            "target_column": self._meta.get("target_column"),
            "random_seed": self._meta.get("random_seed"),
            "model_checkpoint_basename": self._model_path.name,
            "training_config_basename": self._meta_path.name if self._meta_path else None,
        }


def default_service() -> InferenceService:
    svc = InferenceService()
    svc.load()
    return svc
