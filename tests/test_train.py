import numpy as np
import torch

from app.core.config import AppConfig, DataConfig, InferenceConfig, ModelConfig, PathsConfig, TrainingConfig
from app.models.train import compute_pos_weight, train_tabular_classifier


def _tiny_cfg():
    return AppConfig(
        paths=PathsConfig(
            raw_data="x",
            processed_dir="x",
            artifacts_dir="x",
            model_path="x",
            preprocessor_path="x",
            metrics_dir="x",
            reports_dir="x",
            training_config_dump="x",
            history_path="x",
            confusion_matrix_path="x",
        ),
        data=DataConfig(random_seed=0),
        training=TrainingConfig(
            batch_size=8,
            epochs=3,
            early_stopping_patience=5,
            num_workers=0,
        ),
        model=ModelConfig(hidden_layers=[8, 4], dropout=0.0, use_batch_norm=False),
        inference=InferenceConfig(classification_threshold=0.5),
    )


def test_compute_pos_weight():
    y = np.array([0, 0, 0, 1], dtype=np.float32)
    w = compute_pos_weight(y)
    assert w.item() > 0


def test_train_runs_short():
    rng = np.random.default_rng(0)
    n = 60
    d = 12
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (rng.random(n) > 0.7).astype(np.float32)
    split = 40
    cfg = _tiny_cfg()
    model, history, summary = train_tabular_classifier(
        cfg,
        X[:split],
        y[:split],
        X[split:50],
        y[split:50],
        input_dim=d,
        device=torch.device("cpu"),
    )
    assert len(history) >= 1
    assert "best_val_loss" in summary
    assert isinstance(model, torch.nn.Module)
