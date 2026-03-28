"""Training utilities: seeds, device."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
