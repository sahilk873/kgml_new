from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from sklearn.metrics import average_precision_score, roc_auc_score

from kgml_new.config import TrainConfig
from kgml_new.training.history import TrainingHistory
from kgml_new.io.artifacts import torch_load_checkpoint, torch_save_checkpoint


def _setup_amp(device: torch.device, use_amp: bool) -> Optional[torch.cuda.amp.GradScaler]:
    if use_amp and device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None


def _encode(model: nn.Module, data: Data, device: torch.device, edge_aware: bool) -> torch.Tensor:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if edge_aware:
        edge_attr = data.edge_attr.to(device)
        return model(x, edge_index, edge_attr)
    return model(x, edge_index)


def _binary_metrics_from_scores(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> tuple[float, float]:
    pos_np = pos_scores.detach().cpu().numpy()
    neg_np = neg_scores.detach().cpu().numpy()
    y_true = np.concatenate(
        [
            np.ones(pos_np.shape[0], dtype=np.float32),
            np.zeros(neg_np.shape[0], dtype=np.float32),
        ],
        axis=0,
    )
    y_score = np.concatenate([pos_np, neg_np], axis=0)
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def _chunked_dot_product_scores(
    z: torch.Tensor,
    edge_index: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for start in range(0, edge_index.size(1), chunk_size):
        sl = slice(start, min(start + chunk_size, edge_index.size(1)))
        src = edge_index[0, sl]
        dst = edge_index[1, sl]
        scores.append((z[src] * z[dst]).sum(dim=-1).detach().cpu())
    if not scores:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(scores, dim=0)


def train_unsupervised(
    model: nn.Module,
    data: Data,
    train_pos_edge_index: torch.Tensor,
    config: TrainConfig,
    device: torch.device | None = None,
    edge_aware: bool = False,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
) -> tuple[nn.Module, int]:
    """
    Default unsupervised GraphSAGE-style training path.

    Uses neighbor-sampled mini-batch training when the PyG sampling backend is
    available, and falls back to the explicit full-graph trainer otherwise.
    """
    model, last_epoch, _ = train_unsupervised_batched(
        model=model,
        data=data,
        train_pos_edge_index=train_pos_edge_index,
        config=config,
        device=device,
        edge_aware=edge_aware,
        num_neighbors=config.num_neighbors,
        checkpoint_path=checkpoint_path,
        resume=resume,
        save_every_epochs=save_every_epochs,
    )
    return model, last_epoch


def train_unsupervised_fullgraph(
    model: nn.Module,
    data: Data,
    train_pos_edge_index: torch.Tensor,
    config: TrainConfig,
    device: torch.device | None = None,
    edge_aware: bool = False,
    val_pos_edge_index: torch.Tensor | None = None,
    val_neg_edge_index: torch.Tensor | None = None,
    history_path: Path | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
) -> tuple[nn.Module, int, TrainingHistory]:
    return _train_unsupervised_fullgraph(
        model=model,
        data=data,
        train_pos_edge_index=train_pos_edge_index,
        config=config,
        device=device,
        edge_aware=edge_aware,
        val_pos_edge_index=val_pos_edge_index,
        val_neg_edge_index=val_neg_edge_index,
        history_path=history_path,
        checkpoint_path=checkpoint_path,
        resume=resume,
        save_every_epochs=save_every_epochs,
    )


def train_unsupervised_batched(
    model: nn.Module,
    data: Data,
    train_pos_edge_index: torch.Tensor,
    config: TrainConfig,
    device: torch.device | None = None,
    edge_aware: bool = False,
    num_neighbors: list[int] | None = None,
    val_pos_edge_index: torch.Tensor | None = None,
    val_neg_edge_index: torch.Tensor | None = None,
    history_path: Path | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
) -> tuple[nn.Module, int, TrainingHistory]:
    return _train_unsupervised_batched(
        model=model,
        data=data,
        train_pos_edge_index=train_pos_edge_index,
        config=config,
        device=device,
        edge_aware=edge_aware,
        num_neighbors=num_neighbors,
        val_pos_edge_index=val_pos_edge_index,
        val_neg_edge_index=val_neg_edge_index,
        history_path=history_path,
        checkpoint_path=checkpoint_path,
        resume=resume,
        save_every_epochs=save_every_epochs,
    )


def _train_unsupervised_fullgraph(
    model: nn.Module,
    data: Data,
    train_pos_edge_index: torch.Tensor,
    config: TrainConfig,
    device: torch.device | None = None,
    edge_aware: bool = False,
    val_pos_edge_index: torch.Tensor | None = None,
    val_neg_edge_index: torch.Tensor | None = None,
    history_path: Path | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
) -> tuple[nn.Module, int, TrainingHistory]:
    """Full-graph training (no neighbor sampling) with history tracking."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scaler = _setup_amp(device, config.use_amp)
    model = model.to(device)
    data = data.to(device)
    pos = train_pos_edge_index.to(device)
    n = int(data.num_nodes)

    if val_pos_edge_index is not None:
        val_pos_edge_index = val_pos_edge_index.to(device)
    if val_neg_edge_index is not None:
        val_neg_edge_index = val_neg_edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )
    degrees = torch.bincount(data.edge_index[0], minlength=n).float().clamp_min(1.0)
    weights = degrees.pow(0.75)
    weights = weights / weights.sum()
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    start_epoch = 0
    if resume and ckpt_path and ckpt_path.is_file():
        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(
            f"Resuming full-graph training from epoch {start_epoch}"
            f" (checkpoint epoch was {ckpt.get('epoch')})"
        )

    history = TrainingHistory(edge_aware=edge_aware, num_neighbors=None)
    history.config = asdict(config)

    last_epoch = start_epoch - 1
    if start_epoch >= config.epochs:
        print(f"Training already at epoch >= {config.epochs}; skipping training loop.")
    
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, config.epochs):
        model.train()
        perm = torch.randperm(pos.size(1), device=device)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, pos.size(1), config.batch_size):
            sl = perm[start : start + config.batch_size]
            s, d = pos[0, sl], pos[1, sl]
            if s.numel() == 0:
                continue
            
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
                z = _encode(model, data, device, edge_aware)
                pos_logits = (z[s] * z[d]).sum(dim=-1)
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits), reduction="sum"
                )

                neg = torch.multinomial(
                    weights, num_samples=s.size(0) * config.neg_samples, replacement=True
                )
                neg = neg.view(s.size(0), config.neg_samples)

                src_exp = s.unsqueeze(1).expand_as(neg)
                neg_logits = (z[src_exp] * z[neg]).sum(dim=-1)
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits), reduction="sum"
                )

                batch_loss = (pos_loss + neg_loss) / max(s.numel(), 1)

            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()
            
            total_loss += float(batch_loss.detach())
            num_batches += 1
            
            del z, pos_logits, neg_logits, batch_loss

        avg_loss = total_loss / max(num_batches, 1)
        last_epoch = epoch

        history.epoch.append(epoch)
        history.train_loss.append(avg_loss)
        history.learning_rate.append(optimizer.param_groups[0]["lr"])
        history.batch_count.append(num_batches)

        val_auc, val_ap = None, None
        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            with torch.inference_mode():
                z_eval = _encode(model, data, device, edge_aware)
                pos_scores = _chunked_dot_product_scores(
                    z_eval, val_pos_edge_index, chunk_size=max(config.batch_size, 1)
                )
                neg_scores = _chunked_dot_product_scores(
                    z_eval, val_neg_edge_index, chunk_size=max(config.batch_size, 1)
                )
            val_auc, val_ap = _binary_metrics_from_scores(pos_scores, neg_scores)
            history.val_auc.append(val_auc)
            history.val_ap.append(val_ap)
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            
            del z_eval, pos_scores, neg_scores

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            val_str = (
                f" val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
                if val_auc is not None and val_ap is not None
                else ""
            )
            lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}"
            print(f"epoch {epoch:04d} loss={avg_loss:.4f}{val_str}{lr_str} device={device}")

        if ckpt_path is not None and save_every_epochs > 0:
            if (epoch + 1) % save_every_epochs == 0 or epoch == config.epochs - 1:
                torch_save_checkpoint(
                    ckpt_path,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    extra={"train_config": asdict(config), "edge_aware": edge_aware},
                )
        
        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {config.early_stop_patience} epochs)")
            break

    if history_path:
        history.save(history_path)
        print(history.summary())

    return model, last_epoch, history


def _train_unsupervised_batched(
    model: nn.Module,
    data: Data,
    train_pos_edge_index: torch.Tensor,
    config: TrainConfig,
    device: torch.device | None = None,
    edge_aware: bool = False,
    num_neighbors: list[int] | None = None,
    val_pos_edge_index: torch.Tensor | None = None,
    val_neg_edge_index: torch.Tensor | None = None,
    history_path: Path | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
) -> tuple[nn.Module, int, TrainingHistory]:
    """
    Batched link prediction using LinkNeighborLoader (original GraphSAGE algorithm).
    Uses mini-batch neighbor sampling to scale to millions of edges.
    """
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

    def _evaluate_batched_edges(
        eval_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        eval_edge_label = torch.ones(eval_edge_index.size(1), dtype=torch.float32)
        eval_loader = LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            edge_label_index=eval_edge_index,
            edge_label=eval_edge_label,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )
        scores: list[torch.Tensor] = []
        for eval_batch in eval_loader:
            eval_batch = eval_batch.to(device)
            if edge_aware:
                z_eval = model(eval_batch.x, eval_batch.edge_index, eval_batch.edge_attr)
            else:
                z_eval = model(eval_batch.x, eval_batch.edge_index)
            src = eval_batch.edge_label_index[0]
            dst = eval_batch.edge_label_index[1]
            batch_scores = (z_eval[src] * z_eval[dst]).sum(dim=-1).detach().cpu()
            scores.append(batch_scores)
            del eval_batch, z_eval, batch_scores
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not scores:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(scores, dim=0)

    if not WITH_PYG_LIB and not WITH_TORCH_SPARSE:
        print(
            "Neighbor sampling backend unavailable (need pyg-lib or torch-sparse); "
            "falling back to full-graph training."
        )
        return _train_unsupervised_fullgraph(
            model=model,
            data=data,
            train_pos_edge_index=train_pos_edge_index,
            config=config,
            device=device,
            edge_aware=edge_aware,
            val_pos_edge_index=val_pos_edge_index,
            val_neg_edge_index=val_neg_edge_index,
            history_path=history_path,
            checkpoint_path=checkpoint_path,
            resume=resume,
            save_every_epochs=save_every_epochs,
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scaler = _setup_amp(device, config.use_amp)

    model = model.to(device)
    pos = train_pos_edge_index.cpu()

    if val_pos_edge_index is not None:
        val_pos_edge_index = val_pos_edge_index.cpu()
    if val_neg_edge_index is not None:
        val_neg_edge_index = val_neg_edge_index.cpu()

    if num_neighbors is None:
        num_neighbors = config.num_neighbors
    if any(n <= 0 for n in num_neighbors):
        raise ValueError(f"num_neighbors must be positive; got {num_neighbors}")
    if len(num_neighbors) != int(getattr(model, "num_layers", len(num_neighbors))):
        raise ValueError(
            "num_neighbors length must match model num_layers; "
            f"got {len(num_neighbors)} neighbors for {getattr(model, 'num_layers', 'unknown')} layers"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    start_epoch = 0
    if resume and ckpt_path and ckpt_path.is_file():
        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(
            f"Resuming batched training from epoch {start_epoch}"
            f" (checkpoint epoch was {ckpt.get('epoch')})"
        )

    edge_labels = torch.ones(pos.size(1), dtype=torch.float32)
    data.edge_label = edge_labels

    loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=pos,
        edge_label=edge_labels,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = TrainingHistory(edge_aware=edge_aware, num_neighbors=num_neighbors)
    history.config = asdict(config)

    last_epoch = start_epoch - 1
    if start_epoch >= config.epochs:
        print(f"Training already at epoch >= {config.epochs}; skipping training loop.")
    
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
                if edge_aware:
                    z = model(batch.x, batch.edge_index, batch.edge_attr)
                else:
                    z = model(batch.x, batch.edge_index)

                pos_edge_label = batch.edge_label
                pos_edge_index = batch.edge_label_index

                pos_src = pos_edge_index[0]
                pos_dst = pos_edge_index[1]

                local_degrees = (
                    torch.bincount(batch.edge_index[0], minlength=z.size(0))
                    .float()
                    .clamp_min(1.0)
                )

                pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=-1)
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_logits, pos_edge_label, reduction="sum"
                )

                weights = local_degrees.pow(0.75)
                weights = weights / weights.sum()
                neg_samples = pos_src.size(0) * config.neg_samples
                neg = torch.multinomial(weights, num_samples=neg_samples, replacement=True)
                neg = neg.view(pos_src.size(0), config.neg_samples)

                src_exp = pos_src.unsqueeze(1).expand_as(neg)
                neg_logits = (z[src_exp] * z[neg]).sum(dim=-1)
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits), reduction="sum"
                )

                batch_loss = (pos_loss + neg_loss) / max(pos_src.size(0), 1)
            
            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

            total_loss += float(batch_loss.detach())
            num_batches += 1
            
            del batch, z, pos_logits, neg_logits, batch_loss

        avg_loss = total_loss / max(num_batches, 1)
        last_epoch = epoch

        history.epoch.append(epoch)
        history.train_loss.append(avg_loss)
        history.learning_rate.append(optimizer.param_groups[0]["lr"])
        history.batch_count.append(num_batches)

        val_auc, val_ap = None, None
        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            model.eval()
            with torch.inference_mode():
                pos_scores = _evaluate_batched_edges(val_pos_edge_index)
                neg_scores = _evaluate_batched_edges(val_neg_edge_index)
            val_auc, val_ap = _binary_metrics_from_scores(pos_scores, neg_scores)
            history.val_auc.append(val_auc)
            history.val_ap.append(val_ap)
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            
            del pos_scores, neg_scores

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            val_str = (
                f" val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
                if val_auc is not None and val_ap is not None
                else f" batches={num_batches}"
            )
            lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}"
            print(f"epoch {epoch:04d} loss={avg_loss:.4f}{val_str}{lr_str} device={device}")

        if ckpt_path is not None and save_every_epochs > 0:
            if (epoch + 1) % save_every_epochs == 0 or epoch == config.epochs - 1:
                torch_save_checkpoint(
                    ckpt_path,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    extra={
                        "train_config": asdict(config),
                        "edge_aware": edge_aware,
                        "num_neighbors": num_neighbors,
                    },
                )
        
        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {config.early_stop_patience} epochs)")
            break

    if history_path:
        history.save(history_path)
        print(history.summary())

    return model, last_epoch, history


@torch.inference_mode()
def compute_node_embeddings(
    model: nn.Module,
    data: Data,
    device: torch.device,
    edge_aware: bool = False,
) -> torch.Tensor:
    model.eval()
    return _encode(model, data.to(device), device, edge_aware)


def create_train_val_split(
    edge_index: torch.Tensor,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edge_index into train and validation sets.

    Args:
        edge_index: Edge indices [2, num_edges]
        val_ratio: Fraction of edges to use for validation
        seed: Random seed

    Returns:
        (train_pos, val_pos, val_neg_edge_index)
        - train_pos: Training positive edges
        - val_pos: Validation positive edges
        - val_neg_edge_index: Validation negative edges [2, num_val]
    """
    from kgml_new.training.splits import create_edge_split

    num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    split = create_edge_split(
        edge_index,
        num_src_nodes=num_nodes,
        val_ratio=val_ratio,
        test_ratio=0.0,
        seed=seed,
        undirected=True,
    )
    return (
        split.train_pos_edge_index,
        split.val_pos_edge_index,
        split.val_neg_edge_index,
    )
