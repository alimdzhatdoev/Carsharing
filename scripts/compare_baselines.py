"""
Сравнение baseline-моделей (LogReg, RF, XGBoost) и ablation-экспериментов на тех же сплитах, что и основной train.

Опционально: короткий MLP на каждом ablation (`--mlp-per-ablation`, долго).
Метрики production MLP подмешиваются из `artifacts/metrics/test_metrics.json`, если файл есть.

Usage:
  python scripts/compare_baselines.py
  python scripts/compare_baselines.py --ablations full wo_context
  python scripts/compare_baselines.py --mlp-per-ablation
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
from app.core.logger import get_logger, setup_logging
from app.data.loader import load_raw_csv
from app.data.preprocessing import split_features_target
from app.data.split import train_val_test_split
from app.models.benchmark import (
    attach_production_mlp_metrics,
    results_to_markdown_table,
    run_ablation_benchmark,
)
from app.models.utils import set_seed
from app.utils.common import get_project_root

logger = get_logger(__name__)

REPORT_TEMPLATE = """# Benchmark: baselines vs ablations

Автогенерация: `scripts/compare_baselines.py`. Полный JSON: `artifacts/metrics/benchmark_results.json`.

## Таблица (test set, порог из `configs/config.yaml`)

{table}

## Примечания

- Модели обучаются на **одной и той же** train-выборке и оцениваются на **одном** test; для каждого ablation препроцессор **заново fit** только на train (без утечки).
- `mlp_production` (если есть) — метрики последнего `scripts/train.py` на полном наборе признаков; сравнение с baselines по F1/ROC-AUC на одном тесте.
- `mlp_short` появляется только с флагом `--mlp-per-ablation` (укороченное обучение для каждого среза признаков).

Интерпретация и выводы — в `docs/09_baselines_and_ablations.md`.
"""


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=None,
        help="Подмножество имён ablation (по умолчанию — все из app/features/ablation.py)",
    )
    parser.add_argument("--no-xgb", action="store_true", help="Не обучать XGBoost")
    parser.add_argument(
        "--mlp-per-ablation",
        action="store_true",
        help="Обучить укороченный MLP на каждом ablation (существенно дольше)",
    )
    parser.add_argument("--skip-production-mlp-row", action="store_true")
    args = parser.parse_args()

    root = get_project_root()
    cfg = merge_env_overrides(load_app_config(args.config, root=root), EnvOverrides())
    cfg = resolved_paths(cfg, root)
    set_seed(cfg.data.random_seed)

    df = load_raw_csv(cfg.paths.raw_data)
    train_df, val_df, test_df = train_val_test_split(
        df,
        cfg.data.target_column,
        cfg.data.test_size,
        cfg.data.val_size,
        cfg.data.random_seed,
    )
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    rows = run_ablation_benchmark(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        ablation_names=args.ablations,
        include_xgboost=not args.no_xgb,
        train_mlp_per_ablation=args.mlp_per_ablation,
    )
    if not args.skip_production_mlp_row:
        rows = attach_production_mlp_metrics(rows, Path(cfg.paths.metrics_dir) / "test_metrics.json")

    metrics_dir = Path(cfg.paths.metrics_dir)
    reports_dir = Path(cfg.paths.reports_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_json = metrics_dir / "benchmark_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", out_json)

    md_path = reports_dir / "benchmark_report.md"
    table = results_to_markdown_table(rows)
    md_path.write_text(REPORT_TEMPLATE.format(table=table), encoding="utf-8")
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
