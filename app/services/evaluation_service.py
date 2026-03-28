"""Оценка качества сохранённой модели и чтение отчётов из artifacts/."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from app.core.config import AppConfig, ModelConfig
from app.core.logger import get_logger, setup_logging
from app.data.dataset import TabularDataset
from app.data.loader import load_raw_csv
from app.data.preprocessing import load_preprocessor, split_features_target, transform_features
from app.models.evaluate import evaluate_model, save_confusion_matrix_plot
from app.models.predict import load_model_for_inference, load_training_meta
from app.models.utils import get_device
logger = get_logger(__name__)


def write_model_report(path: Path, metrics: dict, threshold: float) -> None:
    cm = metrics.get("confusion_matrix", [])
    lines = [
        "# Отчёт по качеству модели (тестовая выборка)",
        "",
        "## Метрики",
        "",
        f"- **Accuracy:** {metrics.get('accuracy', 0):.4f}",
        f"- **Precision:** {metrics.get('precision', 0):.4f}",
        f"- **Recall:** {metrics.get('recall', 0):.4f}",
        f"- **F1:** {metrics.get('f1', 0):.4f}",
        f"- **ROC-AUC:** {metrics.get('roc_auc', 'n/a')}",
        f"- **Порог:** {threshold}",
        "",
        "## Матрица ошибок",
        "",
        f"```\n{cm}\n```",
        "",
        "## Интерпретация для каршеринга",
        "",
        "- **Высокий recall** при умеренной precision: сервис чаще отмечает поездки как проблемные — "
        "меньше пропусков реальных инцидентов, но выше нагрузка на поддержку и риск ложных тревог.",
        "- **Высокий precision** при низком recall: меньше ложных срабатываний, но часть реальных "
        "проблемных поездок остаётся незамеченной — выше операционный и репутационный риск.",
        "- **False positive:** лишняя модерация, неудобство честным пользователям.",
        "- **False negative:** пропущенное мошенничество, нарушение или инцидент.",
        "",
        "## Ограничения",
        "",
        "- Демо-данные синтетические; на проде нужны мониторинг дрейфа и калибровка порога.",
        "- Модель не заменяет правила и человеческую экспертизу.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclass
class EvaluationResult:
    success: bool
    message: str
    metrics: dict[str, Any] | None = None
    threshold: float | None = None


def run_full_evaluation(cfg: AppConfig) -> EvaluationResult:
    """Пересчёт метрик на test_split и обновление отчётов (как scripts/evaluate.py)."""
    setup_logging()
    test_path = Path(cfg.paths.processed_dir) / "test_split.csv"
    if not test_path.exists():
        return EvaluationResult(
            success=False,
            message=f"Нет test_split: {test_path}. Сначала обучите модель.",
        )
    try:
        test_df = load_raw_csv(test_path)
        X_test, y_test = split_features_target(test_df)

        preprocessor = load_preprocessor(Path(cfg.paths.preprocessor_path))
        Xt_test = transform_features(preprocessor, X_test)

        meta = load_training_meta(Path(cfg.paths.training_config_dump))
        model_cfg = ModelConfig.model_validate(meta["model"])
        device = get_device()
        model = load_model_for_inference(Path(cfg.paths.model_path), meta, model_cfg, device=device)

        threshold = float(meta.get("classification_threshold", cfg.inference.classification_threshold))
        loader = DataLoader(
            TabularDataset(Xt_test, y_test),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
        )
        metrics = evaluate_model(model, loader, device, threshold)

        metrics_dir = Path(cfg.paths.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
        for name in ("evaluate_metrics.json", "test_metrics.json"):
            with (metrics_dir / name).open("w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)

        cm = np.array(metrics["confusion_matrix"])
        save_confusion_matrix_plot(
            cm, Path(cfg.paths.confusion_matrix_path).with_name("confusion_matrix_eval.png")
        )

        report_md = Path(cfg.paths.reports_dir) / "model_report.md"
        write_model_report(report_md, metrics, threshold)
        logger.info("Evaluation wrote %s", report_md)

        return EvaluationResult(
            success=True,
            message="Оценка завершена.",
            metrics=serializable,
            threshold=threshold,
        )
    except Exception as e:
        logger.exception("evaluation failed")
        return EvaluationResult(success=False, message=str(e))


def read_test_metrics(cfg: AppConfig) -> dict[str, Any] | None:
    p = Path(cfg.paths.metrics_dir) / "test_metrics.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def read_evaluate_metrics(cfg: AppConfig) -> dict[str, Any] | None:
    p = Path(cfg.paths.metrics_dir) / "evaluate_metrics.json"
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def read_training_history(cfg: AppConfig) -> list[dict[str, Any]] | None:
    p = Path(cfg.paths.history_path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def read_model_report_markdown(cfg: AppConfig) -> str | None:
    p = Path(cfg.paths.reports_dir) / "model_report.md"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def read_classification_report_text(cfg: AppConfig) -> str | None:
    p = Path(cfg.paths.reports_dir) / "classification_report_test.txt"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def confusion_matrix_image_path(cfg: AppConfig) -> Path | None:
    p = Path(cfg.paths.confusion_matrix_path)
    if p.exists():
        return p
    p2 = Path(cfg.paths.reports_dir) / "confusion_matrix_eval.png"
    if p2.exists():
        return p2
    return None
