"""
Train tabular MLP — thin wrapper over app.services.training_service.

Usage from project root:
  python scripts/generate_demo_data.py
  python scripts/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import EnvOverrides, load_app_config, merge_env_overrides, resolved_paths
from app.core.logger import get_logger, setup_logging
from app.services.training_service import run_full_training
from app.utils.common import get_project_root

logger = get_logger(__name__)


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    root = get_project_root()
    cfg = load_app_config(args.config, root=root)
    cfg = merge_env_overrides(cfg, EnvOverrides())
    cfg = resolved_paths(cfg, root)

    result = run_full_training(cfg, capture_logs=True)
    for line in result.log_lines[-50:]:
        logger.info("(train log) %s", line)
    if not result.success:
        logger.error("%s", result.message)
        raise SystemExit(1)
    logger.info("%s", result.message)


if __name__ == "__main__":
    main()
