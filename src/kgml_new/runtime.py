"""Devices and RNG seeds for training scripts."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Make numpy / torch / Python RNG deterministic for a given run."""
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def resolve_device(device: str | None) -> torch.device:
    """Pick a :class:`torch.device`, defaulting to CUDA when available."""
    if device is None or str(device).lower() in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
