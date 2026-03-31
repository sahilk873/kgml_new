from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from kgml_new.config import LinkMLPConfig
from kgml_new.io.artifacts import torch_load_checkpoint, torch_save_checkpoint
from kgml_new.models.link_mlp import LinkPredictionMLP
from kgml_new.training.eval import link_prediction_mlp_torch
from kgml_new.training.history import TrainingHistory


def train_link_mlp(
    z: Tensor,
    pos_edge_index: Tensor,
    config: LinkMLPConfig,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
) -> tuple[LinkPredictionMLP, int]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    model = LinkPredictionMLP(
        z.size(1), hidden_dims=config.hidden_dims, dropout=config.dropout
    ).to(device)
    z = z.to(device)
    pos_edge_index = pos_edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    start_epoch = 0
    last_epoch = -1
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if resume and ckpt_path and ckpt_path.is_file():
        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resuming Link MLP from epoch {start_epoch}")

    last_epoch = start_epoch - 1
    if start_epoch >= config.epochs:
        print(f"Link MLP already at epoch >= {config.epochs}; skipping training loop.")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        perm = torch.randperm(pos_edge_index.size(1), device=device)
        sl = perm[: min(config.batch_size, pos_edge_index.size(1))]

        pos_src, pos_dst = pos_edge_index[0, sl], pos_edge_index[1, sl]
        neg_src = pos_src
        neg_dst = torch.randint(0, z.size(0), (pos_src.size(0),), device=device)

        pos_logit = model(z[pos_src], z[pos_dst])
        neg_logit = model(z[neg_src], z[neg_dst])

        pos_y = torch.ones_like(pos_logit)
        neg_y = torch.zeros_like(neg_logit)
        y = torch.cat([pos_y, neg_y])
        logits = torch.cat([pos_logit, neg_logit])

        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_epoch = epoch
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"MLP epoch {epoch:04d} loss={float(loss.detach()):.4f}")

        if ckpt_path is not None and save_every_epochs > 0:
            if (epoch + 1) % save_every_epochs == 0 or epoch == config.epochs - 1:
                torch_save_checkpoint(
                    ckpt_path,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    extra={"link_mlp_config": asdict(config)},
                )

    return model, last_epoch


def train_link_mlp_with_validation(
    z: Tensor,
    pos_edge_index: Tensor,
    config: LinkMLPConfig,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
    val_pos_edge_index: Tensor | None = None,
    val_neg_edge_index: Tensor | None = None,
) -> tuple[LinkPredictionMLP, int, TrainingHistory]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    model = LinkPredictionMLP(
        z.size(1), hidden_dims=config.hidden_dims, dropout=config.dropout
    ).to(device)
    z = z.to(device)
    pos_edge_index = pos_edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    start_epoch = 0
    last_epoch = -1
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if resume and ckpt_path and ckpt_path.is_file():
        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resuming Link MLP from epoch {start_epoch}")

    history = TrainingHistory(edge_aware=False, num_neighbors=None)
    history.config = asdict(config)

    last_epoch = start_epoch - 1
    if start_epoch >= config.epochs:
        print(f"Link MLP already at epoch >= {config.epochs}; skipping training loop.")

    if val_pos_edge_index is not None:
        val_pos_edge_index = val_pos_edge_index.to(device)
    if val_neg_edge_index is not None:
        val_neg_edge_index = val_neg_edge_index.to(device)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        perm = torch.randperm(pos_edge_index.size(1), device=device)
        sl = perm[: min(config.batch_size, pos_edge_index.size(1))]

        pos_src, pos_dst = pos_edge_index[0, sl], pos_edge_index[1, sl]
        neg_src = pos_src
        neg_dst = torch.randint(0, z.size(0), (pos_src.size(0),), device=device)

        pos_logit = model(z[pos_src], z[pos_dst])
        neg_logit = model(z[neg_src], z[neg_dst])

        pos_y = torch.ones_like(pos_logit)
        neg_y = torch.zeros_like(neg_logit)
        y = torch.cat([pos_y, neg_y])
        logits = torch.cat([pos_logit, neg_logit])

        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_epoch = epoch
        history.epoch.append(epoch)
        history.train_loss.append(float(loss.detach()))
        history.learning_rate.append(config.learning_rate)
        history.batch_count.append(1)

        val_auc, val_ap = None, None
        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            metrics = link_prediction_mlp_torch(
                model, val_pos_edge_index, val_neg_edge_index, z
            )
            val_auc = metrics["roc_auc"]
            val_ap = metrics["average_precision"]
            history.val_auc.append(val_auc)
            history.val_ap.append(val_ap)

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            val_str = (
                f" val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
                if val_auc is not None and val_ap is not None
                else ""
            )
            print(
                f"MLP epoch {epoch:04d} loss={float(loss.detach()):.4f}{val_str}"
            )

        if ckpt_path is not None and save_every_epochs > 0:
            if (epoch + 1) % save_every_epochs == 0 or epoch == config.epochs - 1:
                torch_save_checkpoint(
                    ckpt_path,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    extra={"link_mlp_config": asdict(config)},
                )

    return model, last_epoch, history
