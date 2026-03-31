from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data

from kgml_new.config import Node2VecConfig
from kgml_new.io.artifacts import torch_load_checkpoint, torch_save_checkpoint
from kgml_new.training.eval import link_prediction_dot_product
from kgml_new.training.history import TrainingHistory
from kgml_new.models.lightweight_node2vec import train_lightweight_node2vec


def _node2vec_history_config(cfg: Node2VecConfig) -> dict[str, int | float]:
    return {
        "embedding_dim": cfg.embedding_dim,
        "walk_length": cfg.walk_length,
        "context_size": cfg.context_size,
        "walks_per_node": cfg.walks_per_node,
        "p": cfg.p,
        "q": cfg.q,
        "num_negative_samples": cfg.num_negative_samples,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "seed": cfg.seed,
    }


def train_node2vec_embeddings(
    data: Data,
    cfg: Node2VecConfig,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
) -> tuple[object, Tensor, int]:
    """
    Train Node2Vec-style embeddings: uses PyG ``Node2Vec`` when ``pyg-lib`` or
    ``torch-cluster`` is available; otherwise a lightweight pure-PyTorch skip-gram
    trainer (no extra binary deps).

    Returns ``(model, z, last_epoch_index)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    edge_index = data.edge_index.to(device)
    num_nodes = int(data.num_nodes)

    try:
        from torch_geometric.nn import Node2Vec

        model = Node2Vec(
            edge_index,
            embedding_dim=cfg.embedding_dim,
            walk_length=cfg.walk_length,
            context_size=cfg.context_size,
            walks_per_node=cfg.walks_per_node,
            p=cfg.p,
            q=cfg.q,
            num_negative_samples=cfg.num_negative_samples,
            num_nodes=num_nodes,
            sparse=False,
        ).to(device)
        loader = model.loader(batch_size=cfg.batch_size, shuffle=True)
    except ImportError:
        model, z, last_ep = train_lightweight_node2vec(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=cfg.embedding_dim,
            walk_length=cfg.walk_length,
            context_size=cfg.context_size,
            walks_per_node=cfg.walks_per_node,
            num_negative_samples=cfg.num_negative_samples,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
            device=device,
            use_amp=cfg.use_amp,
            grad_clip_norm=cfg.grad_clip_norm,
            checkpoint_path=checkpoint_path,
            resume=resume,
            save_every_epochs=save_every_epochs,
        )
        return model, z, last_ep

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and cfg.use_amp) else None
    ckpt_path = checkpoint_path
    start_epoch = 0
    if resume and ckpt_path and ckpt_path.is_file():
        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resuming PyG Node2Vec from epoch {start_epoch}")

    last_epoch = start_epoch - 1
    if start_epoch >= cfg.epochs:
        print(f"PyG Node2Vec already at epoch >= {cfg.epochs}; skipping training loop.")
    else:
        model.train()
        for epoch in range(start_epoch, cfg.epochs):
            total_loss = 0.0
            n = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    enabled=scaler is not None,
                ):
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                    optimizer.step()
                total_loss += float(loss.detach())
                n += 1
            last_epoch = epoch
            if epoch % 20 == 0 or epoch == cfg.epochs - 1:
                avg = total_loss / max(n, 1)
                print(f"node2vec(pyg) epoch {epoch:04d} loss={avg:.4f}")

            if ckpt_path is not None and save_every_epochs > 0:
                if (epoch + 1) % save_every_epochs == 0 or epoch == cfg.epochs - 1:
                    torch_save_checkpoint(
                        ckpt_path,
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        extra={"kind": "pyg_node2vec"},
                    )

    model.eval()
    with torch.inference_mode():
        z = model()
    return model, z, last_epoch


def train_node2vec_embeddings_with_validation(
    data: Data,
    cfg: Node2VecConfig,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    save_every_epochs: int = 1,
    val_pos_edge_index: Tensor | None = None,
    val_neg_edge_index: Tensor | None = None,
) -> tuple[object, Tensor, int, TrainingHistory]:
    """Train Node2Vec embeddings with proper GPU optimization and memory management.
    
    FIXED: Added mixed precision training, gradient clipping, learning rate scheduling,
    and memory cleanup between batches.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    edge_index = data.edge_index.to(device)
    num_nodes = int(data.num_nodes)

    if val_pos_edge_index is not None:
        val_pos_edge_index = val_pos_edge_index.to(device)
    if val_neg_edge_index is not None:
        val_neg_edge_index = val_neg_edge_index.to(device)

    history = TrainingHistory(edge_aware=False, num_neighbors=None)
    history.config = _node2vec_history_config(cfg)
    
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and cfg.use_amp) else None

    def record_epoch(epoch: int, avg_loss: float, z_eval: Tensor) -> None:
        history.epoch.append(epoch)
        history.train_loss.append(avg_loss)
        history.learning_rate.append(cfg.learning_rate)
        history.batch_count.append(0)
        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            metrics = link_prediction_dot_product(
                z_eval, val_pos_edge_index, val_neg_edge_index
            )
            history.val_auc.append(metrics["roc_auc"])
            history.val_ap.append(metrics["average_precision"])

    try:
        from torch_geometric.nn import Node2Vec

        model = Node2Vec(
            edge_index,
            embedding_dim=cfg.embedding_dim,
            walk_length=cfg.walk_length,
            context_size=cfg.context_size,
            walks_per_node=cfg.walks_per_node,
            p=cfg.p,
            q=cfg.q,
            num_negative_samples=cfg.num_negative_samples,
            num_nodes=num_nodes,
            sparse=False,
        ).to(device)
        loader = model.loader(batch_size=cfg.batch_size, shuffle=True)
    except ImportError:
        def lightweight_epoch_callback(epoch: int, avg_loss: float, model_obj: object) -> None:
            with torch.inference_mode():
                z_eval = model_obj.embedding.weight.data.detach().clone()
            record_epoch(epoch, avg_loss, z_eval)

        model, z, last_epoch = train_lightweight_node2vec(
            edge_index=edge_index,
            num_nodes=num_nodes,
            embedding_dim=cfg.embedding_dim,
            walk_length=cfg.walk_length,
            context_size=cfg.context_size,
            walks_per_node=cfg.walks_per_node,
            num_negative_samples=cfg.num_negative_samples,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            seed=cfg.seed,
            device=device,
            use_amp=cfg.use_amp,
            grad_clip_norm=cfg.grad_clip_norm,
            checkpoint_path=checkpoint_path,
            resume=resume,
            save_every_epochs=save_every_epochs,
            epoch_callback=lightweight_epoch_callback,
        )
        return model, z, last_epoch, history

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )
    ckpt_path = checkpoint_path
    start_epoch = 0
    if resume and ckpt_path and ckpt_path.is_file():
        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        print(f"Resuming PyG Node2Vec from epoch {start_epoch}")

    last_epoch = start_epoch - 1
    best_val_auc = float("-inf")
    patience_counter = 0
    if start_epoch >= cfg.epochs:
        print(f"PyG Node2Vec already at epoch >= {cfg.epochs}; skipping training loop.")
    else:
        model.train()
        for epoch in range(start_epoch, cfg.epochs):
            total_loss = 0.0
            n = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                    optimizer.step()
                
                total_loss += float(loss.detach())
                n += 1
                
                del loss, pos_rw, neg_rw
            
            last_epoch = epoch
            avg = total_loss / max(n, 1)

            with torch.inference_mode():
                z_eval = model()
            record_epoch(epoch, avg, z_eval)
            
            if history.val_auc:
                scheduler.step(history.val_auc[-1])
                if history.val_auc[-1] > best_val_auc:
                    best_val_auc = history.val_auc[-1]
                    patience_counter = 0
                else:
                    patience_counter += 1

            if epoch % 20 == 0 or epoch == cfg.epochs - 1:
                val_str = ""
                if history.val_auc and history.val_ap:
                    val_str = (
                        f" val_auc={history.val_auc[-1]:.4f}"
                        f" val_ap={history.val_ap[-1]:.4f}"
                    )
                lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}"
                print(f"node2vec(pyg) epoch {epoch:04d} loss={avg:.4f}{val_str}{lr_str}")

            if ckpt_path is not None and save_every_epochs > 0:
                if (epoch + 1) % save_every_epochs == 0 or epoch == cfg.epochs - 1:
                    torch_save_checkpoint(
                        ckpt_path,
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        extra={"kind": "pyg_node2vec"},
                    )
            if patience_counter >= cfg.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {cfg.early_stop_patience} epochs)"
                )
                break

    model.eval()
    with torch.inference_mode():
        z = model()
    return model, z, last_epoch, history
