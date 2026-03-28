"""Сборка и переопределение AppConfig для UI и сервисов."""

from __future__ import annotations

from pathlib import Path

from app.core.config import AppConfig, PathsConfig
from app.utils.common import get_project_root, resolve_path


def set_raw_data_relative(cfg: AppConfig, relative_under_raw: str, root: Path | None = None) -> AppConfig:
    """relative_under_raw — имя файла в data/raw/ или путь вида data/raw/foo.csv от корня."""
    base = root or get_project_root()
    rel = relative_under_raw.replace("\\", "/").lstrip("/")
    if not rel.startswith("data/raw/"):
        rel = f"data/raw/{Path(rel).name}"
    p = PathsConfig.model_validate({**cfg.paths.model_dump(), "raw_data": rel})
    return cfg.model_copy(update={"paths": p})


def apply_training_ui_overrides(
    cfg: AppConfig,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_layers: list[int],
    dropout: float,
    use_batch_norm: bool,
    activation: str,
    classification_threshold: float,
    random_seed: int,
) -> AppConfig:
    act: str = "gelu" if activation == "gelu" else "relu"
    new_training = cfg.training.model_copy(
        update={"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate}
    )
    new_model = cfg.model.model_copy(
        update={
            "hidden_layers": hidden_layers,
            "dropout": dropout,
            "use_batch_norm": use_batch_norm,
            "activation": act,
        }
    )
    new_inf = cfg.inference.model_copy(update={"classification_threshold": classification_threshold})
    new_data = cfg.data.model_copy(update={"random_seed": random_seed})
    return cfg.model_copy(
        update={
            "training": new_training,
            "model": new_model,
            "inference": new_inf,
            "data": new_data,
        }
    )


def parse_hidden_layers(text: str) -> list[int]:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts]


def resolved_cfg(cfg: AppConfig, root: Path | None = None) -> AppConfig:
    from app.core.config import resolved_paths

    return resolved_paths(cfg, root or get_project_root())
