"""
Валидация сырого CSV перед обучением + опциональный JSON-профиль.

Usage (из корня репозитория):
  python scripts/prepare_data.py --input data/raw/trips_demo.csv
  python scripts/prepare_data.py --input data/raw/trips_demo.csv --strict --write-profile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.logger import get_logger, setup_logging
from app.data.pipeline import run_prepare_stage
from app.utils.common import get_project_root

logger = get_logger(__name__)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Validate raw training CSV")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce demo categorical vocabularies (see app/data/schema.py)",
    )
    parser.add_argument(
        "--write-profile",
        action="store_true",
        help="Write data/processed/data_profile.json (after successful validation)",
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="data/processed/data_profile.json",
        help="Profile JSON path (relative to project root unless absolute)",
    )
    args = parser.parse_args()
    root = get_project_root()
    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = root / in_path

    prof_path = None
    if args.write_profile:
        prof_path = Path(args.profile_output)
        if not prof_path.is_absolute():
            prof_path = root / prof_path

    _, vr = run_prepare_stage(
        in_path,
        strict_categories=args.strict,
        profile_path=prof_path,
    )

    for w in vr.warnings:
        logger.warning("%s", w)
    if not vr.ok:
        for e in vr.errors:
            logger.error("%s", e)
        return 1

    logger.info(
        "Validation OK: rows=%s target_pos_rate=%s",
        vr.row_count,
        f"{vr.target_positive_rate:.4f}" if vr.target_positive_rate is not None else "n/a",
    )
    if prof_path:
        logger.info("Profile written to %s", prof_path)
    print(json.dumps({"ok": True, "rows": vr.row_count, "target_positive_rate": vr.target_positive_rate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
