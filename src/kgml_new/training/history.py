from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kgml_new.io.artifacts import pickle_save


@dataclass
class TrainingHistory:
    edge_aware: bool
    num_neighbors: list[int] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    batch_count: list[int] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)
    val_ap: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        pickle_save(self.to_dict(), path)

    def summary(self) -> str:
        last_epoch = self.epoch[-1] if self.epoch else None
        last_loss = self.train_loss[-1] if self.train_loss else None
        best_auc = max(self.val_auc) if self.val_auc else None
        best_ap = max(self.val_ap) if self.val_ap else None
        return (
            f"TrainingHistory(last_epoch={last_epoch}, last_loss={last_loss}, "
            f"best_val_auc={best_auc}, best_val_ap={best_ap})"
        )
