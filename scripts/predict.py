"""
CLI batch inference from JSON lines or JSON array file.

Usage:
  python scripts/predict.py --input examples/batch_predict_request.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import EnvOverrides, load_app_config, merge_env_overrides, resolved_paths
from app.services.inference_service import InferenceService
from app.utils.common import get_project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="JSON file: {items: [...]} or list")
    args = parser.parse_args()
    root = get_project_root()
    cfg = merge_env_overrides(load_app_config(root=root), EnvOverrides())
    cfg = resolved_paths(cfg, root)
    svc = InferenceService(cfg)
    svc.load()
    if not svc.ready:
        raise SystemExit("Artifacts missing — run training first.")

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        rows = raw["items"]
    elif isinstance(raw, list):
        rows = raw
    else:
        raise SystemExit("Input must be a list of objects or {\"items\": [...]}")

    preds = svc.predict_batch(rows)
    print(json.dumps(preds, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
