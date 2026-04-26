from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from kgml_new.config import TrainConfig
from kgml_new.models.edge_aware_sage import (
    RelationBasisMixtureSAGELayer,
    mixture_weights_from_relation_indices,
)
from kgml_new.training.history import TrainingHistory
from kgml_new.io.artifacts import torch_load_checkpoint, torch_save_checkpoint

_LOG = logging.getLogger("kgml_new.training.link_unsupervised")


def _build_canonical_edge_relation_lookup(
    train_pos_edge_index: torch.Tensor,
    train_pos_edge_attr: torch.Tensor,
) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    ei = train_pos_edge_index.cpu().long()
    ea = train_pos_edge_attr.cpu().long()
    for i in range(ei.size(1)):
        s, d = int(ei[0, i]), int(ei[1, i])
        a, b = min(s, d), max(s, d)
        lookup[(a, b)] = int(ea[i].item())
    return lookup


def _model_has_basis_mixture(model: nn.Module) -> bool:
    return any(isinstance(m, RelationBasisMixtureSAGELayer) for m in model.modules())


def _relation_ids_for_global_supervision_edges(
    g_src: torch.Tensor,
    g_dst: torch.Tensor,
    lookup: dict[tuple[int, int], int],
    device: torch.device,
) -> torch.Tensor:
    gs = g_src.detach().cpu().tolist()
    gd = g_dst.detach().cpu().tolist()
    ids: list[int] = []
    for i in range(len(gs)):
        a, b = min(gs[i], gd[i]), max(gs[i], gd[i])
        rid = lookup.get((a, b), -1)
        ids.append(rid)
    return torch.tensor(ids, device=device, dtype=torch.long)


def semantic_alignment_loss(
    mixture_weights: torch.Tensor,
    relation_indices: torch.Tensor,
    similarity_matrix: torch.Tensor,
    lambda_reg: float = 0.1,
) -> torch.Tensor:
    """
    Semantic alignment regularizer for basis mixture model.

    Encourages the learned mixture weights to preserve the semantic geometry
    from the frozen relation embeddings.

    Args:
        mixture_weights: (batch_size, num_bases) mixture weights from basis mixture
        relation_indices: (batch_size,) indices of relations for each edge
        similarity_matrix: (num_relations, num_relations) cosine similarity matrix
        lambda_reg: weight for the regularization term

    Loss: sum_{i,j} sim(semantic_i, semantic_j) * ||alpha_i - alpha_j||^2
    """
    batch_size = mixture_weights.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=mixture_weights.device)

    rel_idx_i = relation_indices.unsqueeze(1)
    rel_idx_j = relation_indices.unsqueeze(0)

    sem_sim = similarity_matrix[rel_idx_i, rel_idx_j]

    alpha_i = mixture_weights.unsqueeze(1)
    alpha_j = mixture_weights.unsqueeze(0)
    diff = (alpha_i - alpha_j).pow(2).sum(dim=-1)

    alignment_loss = (sem_sim * diff).sum() / (batch_size * batch_size)

    return lambda_reg * alignment_loss


def _encode(
    model: nn.Module, data: Data, device: torch.device, edge_aware: bool
) -> torch.Tensor:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if edge_aware:
        edge_attr = data.edge_attr.to(device)
        return model(x, edge_index, edge_attr)
    return model(x, edge_index)


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
        alignment_train_pos_edge_index=None,
        alignment_train_pos_edge_attr=None,
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
    semantic_similarity_matrix: torch.Tensor | None = None,
    semantic_alignment_lambda: float = 0.0,
    edge_attr_for_alignment: torch.Tensor | None = None,
    alignment_train_pos_edge_index: torch.Tensor | None = None,
    alignment_train_pos_edge_attr: torch.Tensor | None = None,
    relation_decode_bundle: nn.Module | None = None,
    train_uv_relation_lookup: dict[tuple[int, int], int] | None = None,
    metric_eval_data: Data | None = None,
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
        semantic_similarity_matrix=semantic_similarity_matrix,
        semantic_alignment_lambda=semantic_alignment_lambda,
        edge_attr_for_alignment=edge_attr_for_alignment,
        alignment_train_pos_edge_index=alignment_train_pos_edge_index,
        alignment_train_pos_edge_attr=alignment_train_pos_edge_attr,
        relation_decode_bundle=relation_decode_bundle,
        train_uv_relation_lookup=train_uv_relation_lookup,
        metric_eval_data=metric_eval_data,
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
    semantic_similarity_matrix: torch.Tensor | None = None,
    semantic_alignment_lambda: float = 0.0,
    edge_attr_for_alignment: torch.Tensor | None = None,
    relation_decode_bundle: nn.Module | None = None,
    train_uv_relation_lookup: dict[tuple[int, int], int] | None = None,
    metric_eval_data: Data | None = None,
) -> tuple[nn.Module, int, TrainingHistory]:
    """Full-graph training (no neighbor sampling) with history tracking."""
    from kgml_new.training.eval import (
        dot_product_link_logits,
        evaluate_inductive_link_prediction,
        link_prediction_dot_product,
        mean_bce_logits_link_prediction,
    )
    from kgml_new.training.relation_decode_batch import relation_ids_for_uv_pairs

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)
    pos = train_pos_edge_index.to(device)
    n = int(data.num_nodes)

    if val_pos_edge_index is not None:
        val_pos_edge_index = val_pos_edge_index.to(device)
    if val_neg_edge_index is not None:
        val_neg_edge_index = val_neg_edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
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
        _LOG.info(
            "Resuming full-graph training from epoch %s (checkpoint epoch was %s)",
            start_epoch,
            ckpt.get("epoch"),
        )

    history = TrainingHistory(edge_aware=edge_aware, num_neighbors=None)
    history.config = asdict(config)

    last_epoch = start_epoch - 1
    if start_epoch >= config.epochs:
        _LOG.info(
            "Training already at epoch >= %s; skipping training loop.", config.epochs
        )

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
            optimizer.zero_grad()
            z = _encode(model, data, device, edge_aware)
            if relation_decode_bundle is not None and train_uv_relation_lookup is not None:
                pos_ei = torch.stack([s, d], dim=0)
                pos_rel = relation_ids_for_uv_pairs(
                    s, d, train_uv_relation_lookup, device=device
                )
                pos_logits = relation_decode_bundle.decode_logits(
                    z, pos_ei, pos_rel
                )
                neg = torch.multinomial(
                    weights,
                    num_samples=s.size(0) * config.neg_samples,
                    replacement=True,
                )
                neg = neg.view(s.size(0), config.neg_samples)
                src_exp = s.unsqueeze(1).expand_as(neg)
                neg_ei = torch.stack([src_exp.reshape(-1), neg.reshape(-1)], dim=0)
                neg_rel = pos_rel.repeat_interleave(int(config.neg_samples))
                neg_logits = relation_decode_bundle.decode_logits(
                    z, neg_ei, neg_rel
                ).view(s.size(0), config.neg_samples)
            else:
                pos_logits = (z[s] * z[d]).sum(dim=-1)
                neg = torch.multinomial(
                    weights,
                    num_samples=s.size(0) * config.neg_samples,
                    replacement=True,
                )
                neg = neg.view(s.size(0), config.neg_samples)
                src_exp = s.unsqueeze(1).expand_as(neg)
                neg_logits = (z[src_exp] * z[neg]).sum(dim=-1)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="sum"
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="sum"
            )

            batch_loss = (pos_loss + neg_loss) / max(s.numel(), 1)

            if (
                semantic_alignment_lambda > 0
                and semantic_similarity_matrix is not None
                and edge_attr_for_alignment is not None
                and edge_aware
                and _model_has_basis_mixture(model)
            ):
                edge_attrs = edge_attr_for_alignment.to(device)
                edge_rel_indices = (
                    edge_attrs[sl] if edge_attrs.size(0) >= pos.size(1) else edge_attrs
                )

                alpha = mixture_weights_from_relation_indices(
                    model, edge_rel_indices.to(device)
                )
                if alpha.size(0) > 0:
                    align_loss = semantic_alignment_loss(
                        alpha,
                        edge_rel_indices.to(device),
                        semantic_similarity_matrix.to(device),
                        lambda_reg=semantic_alignment_lambda,
                    )
                    batch_loss = batch_loss + align_loss

            batch_loss.backward()
            optimizer.step()
            total_loss += float(batch_loss.detach())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        last_epoch = epoch

        history.epoch.append(epoch)
        history.train_loss.append(avg_loss)
        history.learning_rate.append(config.learning_rate)
        history.batch_count.append(num_batches)

        val_auc, val_ap, val_loss = None, None, None
        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            if (
                relation_decode_bundle is not None
                and metric_eval_data is not None
            ):
                metrics = evaluate_inductive_link_prediction(
                    model,
                    metric_eval_data,
                    pos_edge_index=val_pos_edge_index,
                    neg_edge_index=val_neg_edge_index,
                    negatives_per_pos=int(config.neg_samples),
                    device=device,
                    edge_aware=edge_aware,
                    num_neighbors=list(config.num_neighbors),
                    decoder="dot",
                    decoder_model=None,
                    batch_size=int(config.batch_size),
                    rel_decode_bundle=relation_decode_bundle,
                    return_scores=True,
                )
                pos_s = torch.from_numpy(metrics["pos_scores"]).float()
                neg_s = torch.from_numpy(metrics["neg_scores"]).float()
                val_loss = float(mean_bce_logits_link_prediction(pos_s, neg_s))
            else:
                with torch.inference_mode():
                    z_eval = _encode(model, data, device, edge_aware)
                metrics = link_prediction_dot_product(
                    z_eval, val_pos_edge_index, val_neg_edge_index
                )
                pos_l, neg_l = dot_product_link_logits(
                    z_eval, val_pos_edge_index, val_neg_edge_index
                )
                val_loss = float(mean_bce_logits_link_prediction(pos_l, neg_l))
            val_auc = metrics["roc_auc"]
            val_ap = metrics["average_precision"]
            history.val_auc.append(val_auc)
            history.val_ap.append(val_ap)
            history.val_loss.append(val_loss)

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            val_str = (
                f" val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
                if val_auc is not None and val_ap is not None
                else ""
            )
            if val_loss is not None:
                val_str = f"{val_str} val_loss={val_loss:.4f}".strip()
            _LOG.info(
                "epoch %04d loss=%.4f%s device=%s",
                epoch,
                avg_loss,
                val_str,
                device,
            )

        if ckpt_path is not None and save_every_epochs > 0:
            if (epoch + 1) % save_every_epochs == 0 or epoch == config.epochs - 1:
                torch_save_checkpoint(
                    ckpt_path,
                    epoch=epoch,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    extra={"train_config": asdict(config), "edge_aware": edge_aware},
                )

    if history_path:
        history.save(history_path)
        _LOG.info("%s", history.summary())

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
    semantic_similarity_matrix: torch.Tensor | None = None,
    semantic_alignment_lambda: float = 0.0,
    edge_attr_for_alignment: torch.Tensor | None = None,
    alignment_train_pos_edge_index: torch.Tensor | None = None,
    alignment_train_pos_edge_attr: torch.Tensor | None = None,
    relation_decode_bundle: nn.Module | None = None,
    train_uv_relation_lookup: dict[tuple[int, int], int] | None = None,
    metric_eval_data: Data | None = None,
) -> tuple[nn.Module, int, TrainingHistory]:
    """
    Batched link prediction using LinkNeighborLoader (original GraphSAGE algorithm).
    Uses mini-batch neighbor sampling to scale to millions of edges.
    """
    from kgml_new.training.eval import (
        dot_product_link_logits,
        evaluate_inductive_link_prediction,
        link_prediction_dot_product,
        mean_bce_logits_link_prediction,
    )
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

    if not WITH_PYG_LIB and not WITH_TORCH_SPARSE:
        _LOG.warning(
            "Neighbor sampling backend unavailable; falling back to full-graph training."
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
            semantic_similarity_matrix=semantic_similarity_matrix,
            semantic_alignment_lambda=semantic_alignment_lambda,
            edge_attr_for_alignment=edge_attr_for_alignment,
            relation_decode_bundle=relation_decode_bundle,
            train_uv_relation_lookup=train_uv_relation_lookup,
            metric_eval_data=metric_eval_data,
        )

    alignment_lookup: dict[tuple[int, int], int] | None = None
    alignment_batched_skip_warned = False
    if (
        semantic_alignment_lambda > 0
        and semantic_similarity_matrix is not None
        and edge_aware
    ):
        if (
            alignment_train_pos_edge_index is not None
            and alignment_train_pos_edge_attr is not None
        ):
            alignment_lookup = _build_canonical_edge_relation_lookup(
                alignment_train_pos_edge_index,
                alignment_train_pos_edge_attr,
            )
        else:
            if not alignment_batched_skip_warned:
                _LOG.warning(
                    "semantic_alignment_lambda > 0 but alignment_train_pos_edge_index/"
                    "alignment_train_pos_edge_attr not provided; skipping alignment in "
                    "batched training."
                )
                alignment_batched_skip_warned = True

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    pos = train_pos_edge_index.cpu()

    if val_pos_edge_index is not None:
        val_pos_edge_index = val_pos_edge_index.to(device)
    if val_neg_edge_index is not None:
        val_neg_edge_index = val_neg_edge_index.to(device)

    if num_neighbors is None:
        num_neighbors = config.num_neighbors

    from kgml_new.training.relation_decode_batch import (
        relation_ids_for_link_neighbor_batch,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    start_epoch = 0
    if resume and ckpt_path and ckpt_path.is_file():
        ckpt = torch_load_checkpoint(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        _LOG.info(
            "Resuming batched training from epoch %s (checkpoint epoch was %s)",
            start_epoch,
            ckpt.get("epoch"),
        )

    edge_labels = torch.ones(pos.size(1), dtype=torch.float32)

    loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=pos,
        edge_label=edge_labels,
        neg_sampling_ratio=float(config.neg_samples),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = TrainingHistory(edge_aware=edge_aware, num_neighbors=num_neighbors)
    history.config = asdict(config)

    last_epoch = start_epoch - 1
    if start_epoch >= config.epochs:
        _LOG.info(
            "Training already at epoch >= %s; skipping training loop.", config.epochs
        )

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if edge_aware:
                z = model(batch.x, batch.edge_index, batch.edge_attr)
            else:
                z = model(batch.x, batch.edge_index)

            edge_label_index = batch.edge_label_index
            edge_label = batch.edge_label.float()
            if relation_decode_bundle is not None and train_uv_relation_lookup is not None:
                rel_ids = relation_ids_for_link_neighbor_batch(
                    edge_label_index,
                    edge_label,
                    batch.n_id,
                    train_uv_relation_lookup,
                    int(config.neg_samples),
                    device=device,
                )
                logits = relation_decode_bundle.decode_logits(
                    z, edge_label_index, rel_ids
                )
            else:
                logits = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
            batch_loss = F.binary_cross_entropy_with_logits(logits, edge_label)

            if (
                semantic_alignment_lambda > 0
                and semantic_similarity_matrix is not None
                and alignment_lookup is not None
                and edge_aware
                and _model_has_basis_mixture(model)
            ):
                pos_mask = edge_label > 0.5
                if pos_mask.any():
                    loc_src = batch.edge_label_index[0, pos_mask]
                    loc_dst = batch.edge_label_index[1, pos_mask]
                    g_src = batch.n_id[loc_src]
                    g_dst = batch.n_id[loc_dst]
                    rel_idx = _relation_ids_for_global_supervision_edges(
                        g_src, g_dst, alignment_lookup, device
                    )
                    valid = rel_idx >= 0
                    if valid.sum() >= 2:
                        rel_kept = rel_idx[valid]
                        alpha = mixture_weights_from_relation_indices(model, rel_kept)
                        align_loss = semantic_alignment_loss(
                            alpha,
                            rel_kept,
                            semantic_similarity_matrix.to(device),
                            lambda_reg=semantic_alignment_lambda,
                        )
                        batch_loss = batch_loss + align_loss

            batch_loss.backward()
            optimizer.step()

            total_loss += float(batch_loss.detach())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        last_epoch = epoch

        history.epoch.append(epoch)
        history.train_loss.append(avg_loss)
        history.learning_rate.append(config.learning_rate)
        history.batch_count.append(num_batches)

        val_auc, val_ap, val_loss = None, None, None
        if val_pos_edge_index is not None and val_neg_edge_index is not None:
            if (
                relation_decode_bundle is not None
                and metric_eval_data is not None
            ):
                metrics = evaluate_inductive_link_prediction(
                    model,
                    metric_eval_data,
                    pos_edge_index=val_pos_edge_index,
                    neg_edge_index=val_neg_edge_index,
                    negatives_per_pos=int(config.neg_samples),
                    device=device,
                    edge_aware=edge_aware,
                    num_neighbors=num_neighbors,
                    decoder="dot",
                    decoder_model=None,
                    batch_size=int(config.batch_size),
                    rel_decode_bundle=relation_decode_bundle,
                    return_scores=True,
                )
                pos_s = torch.from_numpy(metrics["pos_scores"]).float()
                neg_s = torch.from_numpy(metrics["neg_scores"]).float()
                val_loss = float(mean_bce_logits_link_prediction(pos_s, neg_s))
            else:
                with torch.inference_mode():
                    z_eval = _encode(model, data, device, edge_aware)
                metrics = link_prediction_dot_product(
                    z_eval, val_pos_edge_index, val_neg_edge_index
                )
                pos_l, neg_l = dot_product_link_logits(
                    z_eval, val_pos_edge_index, val_neg_edge_index
                )
                val_loss = float(mean_bce_logits_link_prediction(pos_l, neg_l))
            val_auc = metrics["roc_auc"]
            val_ap = metrics["average_precision"]
            history.val_auc.append(val_auc)
            history.val_ap.append(val_ap)
            history.val_loss.append(val_loss)

        if epoch % 10 == 0 or epoch == config.epochs - 1:
            val_str = (
                f" val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
                if val_auc is not None and val_ap is not None
                else f" batches={num_batches}"
            )
            if val_loss is not None:
                val_str = f"{val_str} val_loss={val_loss:.4f}".strip()
            _LOG.info(
                "epoch %04d loss=%.4f%s device=%s",
                epoch,
                avg_loss,
                val_str,
                device,
            )

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

    if history_path:
        history.save(history_path)
        _LOG.info("%s", history.summary())

    return model, last_epoch, history


def compute_node_embeddings(
    model: nn.Module,
    data: Data,
    device: torch.device,
    edge_aware: bool = False,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
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
