import numpy as np

from app.models.evaluate import binary_metrics


def test_binary_metrics_includes_curves_and_counts():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.binomial(1, 0.25, size=n).astype(np.int64)
    logits = rng.normal(size=n) + 0.8 * (y * 2 - 1)
    m = binary_metrics(y, logits, threshold=0.5)
    assert m["n_test"] == n
    assert m["n_positive"] + m["n_negative"] == n
    assert m["roc_curve"] is not None
    assert len(m["roc_curve"]["fpr"]) == len(m["roc_curve"]["tpr"])
    assert m["pr_curve"] is not None
    assert len(m["pr_curve"]["precision"]) == len(m["pr_curve"]["recall"])
