from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def pickle_save(obj: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def torch_save_checkpoint(
    path: Path,
    epoch: int,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
    }
    if optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = optimizer_state_dict
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def torch_load_checkpoint(
    path: Path,
    map_location: str | None = None,
) -> dict[str, Any]:
    if map_location is not None:
        return torch.load(path, map_location=map_location)
    return torch.load(path)


def torch_save_tensor(tensor: torch.Tensor, path: Path) -> None:
    torch.save(tensor, path)


def torch_load_tensor(path: Path, map_location: str | None = None) -> torch.Tensor:
    if map_location is not None:
        return torch.load(path, map_location=map_location)
    return torch.load(path)