import numpy as np

from app.models.baselines import fit_predict_baselines
from app.models.evaluate import binary_metrics_from_proba


def test_binary_metrics_from_proba_perfect():
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    p = np.array([0.1, 0.2, 0.9, 0.8], dtype=np.float32)
    m = binary_metrics_from_proba(y, p, 0.5)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0


def test_baselines_fit_smoke():
    rng = np.random.default_rng(0)
    n, d = 80, 8
    X_train = rng.normal(size=(n, d)).astype(np.float32)
    y_train = (rng.random(n) > 0.75).astype(np.int64)
    X_test = rng.normal(size=(30, d)).astype(np.float32)
    y_test = (rng.random(30) > 0.75).astype(np.int64)
    res = fit_predict_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=0.5,
        random_state=0,
        include_xgboost=False,
    )
    assert len(res) == 2
    assert res[0].name == "logistic_regression"
    assert res[0].metrics["accuracy"] is not None


def test_build_preprocessor_subset():
    import pandas as pd

    from app.data.preprocessing import (
        build_preprocessor_for_columns,
        fit_preprocessor,
        transform_features,
    )

    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan],
            "b": ["x", "y", "x"],
        }
    )
    pre = build_preprocessor_for_columns(["a"], ["b"])
    fit_preprocessor(pre, df)
    out = transform_features(pre, df)
    assert out.shape[0] == 3
    assert out.shape[1] >= 2
