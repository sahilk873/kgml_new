from __future__ import annotations

import pickle
from pathlib import Path

import torch


def torch_save_checkpoint(
    path: Path,
    epoch: int,
    model_state_dict: dict,
    optimizer_state_dict: dict | None = None,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
    }
    if optimizer_state_dict is not None:
        ckpt["optimizer_state_dict"] = optimizer_state_dict
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def torch_load_checkpoint(
    path: Path,
    map_location: torch.device | None = None,
) -> dict:
    return torch.load(path, map_location=map_location)


def pickle_load(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
