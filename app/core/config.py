"""YAML config + optional .env overrides via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.common import get_project_root, resolve_path


class PathsConfig(BaseModel):
    raw_data: str
    processed_dir: str
    artifacts_dir: str
    model_path: str
    preprocessor_path: str
    metrics_dir: str
    reports_dir: str
    training_config_dump: str
    history_path: str
    confusion_matrix_path: str


class DataConfig(BaseModel):
    target_column: str = "target_class"
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42


class TrainingConfig(BaseModel):
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    num_workers: int = 0


class ModelConfig(BaseModel):
    hidden_layers: list[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: Literal["relu", "gelu"] = "relu"


class InferenceConfig(BaseModel):
    classification_threshold: float = 0.5


class AppConfig(BaseModel):
    paths: PathsConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    inference: InferenceConfig


class EnvOverrides(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: str | None = Field(default=None, validation_alias="PROJECT_ROOT")
    artifacts_dir: str | None = Field(default=None, validation_alias="ARTIFACTS_DIR")
    model_path: str | None = Field(default=None, validation_alias="MODEL_PATH")
    preprocessor_path: str | None = Field(default=None, validation_alias="PREPROCESSOR_PATH")
    training_config_path: str | None = Field(default=None, validation_alias="TRAINING_CONFIG_PATH")
    classification_threshold: float | None = Field(default=None, validation_alias="CLASSIFICATION_THRESHOLD")


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_config(
    config_path: str | Path | None = None,
    root: Path | None = None,
) -> AppConfig:
    base = root or get_project_root()
    cfg_file = Path(config_path) if config_path else base / "configs" / "config.yaml"
    if not cfg_file.is_absolute():
        cfg_file = (base / cfg_file).resolve()
    data = load_yaml_config(cfg_file)
    return AppConfig.model_validate(data)


def merge_env_overrides(cfg: AppConfig, env: EnvOverrides | None = None) -> AppConfig:
    env = env or EnvOverrides()
    dump = cfg.model_dump()
    if env.artifacts_dir:
        ad = env.artifacts_dir
        dump["paths"]["artifacts_dir"] = ad
        # keep default relative layout under artifacts if only dir overridden
    if env.model_path:
        dump["paths"]["model_path"] = env.model_path
    if env.preprocessor_path:
        dump["paths"]["preprocessor_path"] = env.preprocessor_path
    if env.training_config_path:
        dump["paths"]["training_config_dump"] = env.training_config_path
    if env.classification_threshold is not None:
        dump["inference"]["classification_threshold"] = env.classification_threshold
    return AppConfig.model_validate(dump)


def resolved_paths(cfg: AppConfig, root: Path | None = None) -> AppConfig:
    """Return new AppConfig with all paths resolved to absolute."""
    base = root or get_project_root()
    p = cfg.paths.model_dump()
    for key, val in p.items():
        p[key] = str(resolve_path(val, base))
    new = cfg.model_copy(update={"paths": PathsConfig.model_validate(p)})
    return new
