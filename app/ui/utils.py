"""Общие хелперы для Streamlit-страниц."""

from __future__ import annotations

from app.core.config import AppConfig, EnvOverrides, load_app_config, merge_env_overrides, resolved_paths
from app.utils.common import get_project_root


def get_resolved_config() -> AppConfig:
    root = get_project_root()
    cfg = merge_env_overrides(load_app_config(root=root), EnvOverrides())
    return resolved_paths(cfg, root)
