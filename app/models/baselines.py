"""
Классические baseline-модели на том же преобразованном табличном входе, что и MLP.

Используются для честного сравнения: одни и те же train/test и порог классификации.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.models.evaluate import binary_metrics_from_proba


@dataclass
class BaselineResult:
    name: str
    metrics: dict[str, Any]
    estimator: object | None = None


def _scale_pos_weight(y: np.ndarray) -> float:
    y = y.astype(np.int64)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return float(n_neg / n_pos)


def fit_predict_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float,
    random_state: int,
    include_xgboost: bool = True,
) -> list[BaselineResult]:
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    results: list[BaselineResult] = []

    lr = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_test)[:, 1]
    results.append(
        BaselineResult(
            "logistic_regression",
            binary_metrics_from_proba(y_test, p_lr, threshold),
            lr,
        )
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    results.append(
        BaselineResult(
            "random_forest",
            binary_metrics_from_proba(y_test, p_rf, threshold),
            rf,
        )
    )

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            results.append(
                BaselineResult(
                    "xgboost",
                    {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "roc_auc": None,
                        "note": "xgboost not installed",
                    },
                    None,
                )
            )
        else:
            spw = _scale_pos_weight(y_train)
            xgb = XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1,
                scale_pos_weight=spw,
                eval_metric="logloss",
            )
            xgb.fit(X_train, y_train)
            p_x = xgb.predict_proba(X_test)[:, 1]
            results.append(
                BaselineResult(
                    "xgboost",
                    binary_metrics_from_proba(y_test, p_x, threshold),
                    xgb,
                )
            )

    return results


def metrics_summary(result: BaselineResult) -> dict[str, Any]:
    m = result.metrics
    return {
        "model": result.name,
        "accuracy": m.get("accuracy"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "f1": m.get("f1"),
        "roc_auc": m.get("roc_auc"),
    }