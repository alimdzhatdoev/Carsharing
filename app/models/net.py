"""Configurable MLP for binary tabular classification (single logit output)."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.core.config import ModelConfig


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class TabularMLP(nn.Module):
    """
    Binary classifier: forward returns logits of shape (batch, 1).
    Use BCEWithLogitsLoss; for probability apply sigmoid to logits.
    """

    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        act = _activation(cfg.activation)
        for i, hidden in enumerate(cfg.hidden_layers):
            layers.append(nn.Linear(dim, hidden))
            if cfg.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(act)
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
