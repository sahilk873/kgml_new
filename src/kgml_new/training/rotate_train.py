"""Training and evaluation for RotatE link prediction."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from kgml_new.config import RotatEConfig
from kgml_new.models.rotate import RotatE
from kgml_new.training.eval import (
    grouped_link_prediction_metrics,
    mean_bce_logits_link_prediction,
)
from kgml_new.training.history import TrainingHistory
from kgml_new.training.relation_decode_batch import relation_ids_for_query_batch

_LOG = logging.getLogger(__name__)


def build_undirected_edge_set(edge_index: Tensor) -> set[tuple[int, int]]:
    """Canonical undirected keys ``(min(u,v), max(u,v))`` for adjacency membership."""
    ei = edge_index.cpu().long()
    out: set[tuple[int, int]] = set()
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        if u > v:
            u, v = v, u
        out.add((u, v))
    return out


def _sample_tail_corruption(
    src: Tensor,
    true_dst: Tensor,
    *,
    num_nodes: int,
    edge_set: set[tuple[int, int]],
    gen: torch.Generator,
    max_tries: int = 50,
) -> Tensor:
    """Vectorized tail corruption: ``src``, ``true_dst`` shape ``[B]`` -> ``dst_neg`` ``[B]``."""
    device = src.device
    b = int(src.numel())
    out = torch.empty(b, dtype=torch.long, device=device)
    for i in range(b):
        s = int(src[i].item())
        t_true = int(true_dst[i].item())
        for _ in range(max_tries):
            t = int(torch.randint(0, num_nodes, (1,), generator=gen, device=device).item())
            if t == s or t == t_true:
                continue
            a, b_ = (s, t) if s <= t else (t, s)
            if (a, b_) not in edge_set:
                out[i] = t
                break
        else:
            # fall back: any node != s (weak negative)
            t = (s + 1) % num_nodes
            if t == s:
                t = (s + 2) % num_nodes
            out[i] = t
    return out


def evaluate_rotate_link_prediction(
    model: RotatE,
    *,
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    negatives_per_pos: int,
    query_uv_relation: dict[tuple[int, int], int],
    device: torch.device,
    batch_size: int = 4096,
    query_buckets: list[str] | None = None,
    return_scores: bool = False,
) -> dict[str, float | np.ndarray | list | object]:
    """Compute grouped LP metrics using RotatE logits (no graph encoder)."""
    from kgml_new.training.eval import _bucket_metrics

    model.eval()
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)
    n_pos = int(pos_edge_index.size(1))
    if n_pos == 0:
        empty: dict[str, float | list | np.ndarray] = {
            "roc_auc": float("nan"),
            "average_precision": float("nan"),
            "hits@1": float("nan"),
            "hits@3": float("nan"),
            "hits@10": float("nan"),
            "bucket_metrics": _bucket_metrics(
                pos_scores=np.empty((0,), dtype=np.float32),
                neg_scores=np.empty((0, negatives_per_pos), dtype=np.float32),
                query_buckets=query_buckets,
            ),
        }
        if return_scores:
            empty["pos_scores"] = np.empty((0,), dtype=np.float32)
            empty["neg_scores"] = np.empty((0, negatives_per_pos), dtype=np.float32)
        return empty

    eli = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    rel_ids = relation_ids_for_query_batch(
        pos_edge_index,
        neg_edge_index,
        query_uv_relation,
        negatives_per_pos,
        device=device,
    )
    src = eli[0]
    dst = eli[1]
    all_scores: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, int(src.numel()), batch_size):
            end = min(start + batch_size, int(src.numel()))
            all_scores.append(
                model.score_logits(
                    src[start:end], rel_ids[start:end], dst[start:end]
                )
            )
    scores = torch.cat(all_scores, dim=0)
    pos_scores = scores[:n_pos].detach().cpu().numpy()
    neg_scores = scores[n_pos:].view(n_pos, negatives_per_pos).detach().cpu().numpy()
    metrics: dict[str, float | list | np.ndarray | object] = grouped_link_prediction_metrics(
        pos_scores, neg_scores
    )
    metrics["bucket_metrics"] = _bucket_metrics(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        query_buckets=query_buckets,
    )
    if return_scores:
        metrics["pos_scores"] = pos_scores
        metrics["neg_scores"] = neg_scores
    return metrics


def train_rotate_with_validation(
    model: RotatE,
    train_data: Data,
    train_pos_edge_index: Tensor,
    train_pos_edge_attr: Tensor,
    cfg: RotatEConfig,
    device: torch.device,
    *,
    val_pos_edge_index: Tensor,
    val_neg_edge_index: Tensor,
    negatives_per_pos: int,
    query_uv_relation: dict[tuple[int, int], int],
    val_query_buckets: list[str] | None = None,
    test_pos_edge_index: Tensor | None = None,
    test_neg_edge_index: Tensor | None = None,
    test_query_buckets: list[str] | None = None,
    return_scores: bool = False,
) -> tuple[RotatE, int, TrainingHistory, dict, dict | None]:
    """
    Train RotatE with BCE on positive edges + tail corruptions; validate with grouped metrics.
    """
    model = model.to(device)
    num_nodes = int(train_data.num_nodes)
    edge_set = build_undirected_edge_set(train_data.edge_index)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(cfg.seed))

    train_pos_edge_index = train_pos_edge_index.to(device)
    train_pos_edge_attr = train_pos_edge_attr.to(device).long().view(-1)
    n_train = int(train_pos_edge_index.size(1))
    if n_train == 0:
        raise ValueError("RotatE needs at least one training positive edge.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history = TrainingHistory(
        edge_aware=False,
        num_neighbors=None,
        config={
            "embedding_dim": cfg.embedding_dim,
            "batch_size": cfg.batch_size,
            "neg_samples": cfg.neg_samples,
            "gamma": cfg.gamma,
            "weight_decay": cfg.weight_decay,
            "learning_rate": cfg.learning_rate,
        },
    )

    best_auc = float("-inf")
    best_state: dict | None = None
    patience_left = int(cfg.early_stop_patience)
    last_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=gen, device=device)
        epoch_loss = 0.0
        n_batches = 0
        batch_idx = 0
        for start in range(0, n_train, cfg.batch_size):
            end = min(start + cfg.batch_size, n_train)
            idx = perm[start:end]
            src = train_pos_edge_index[0, idx]
            dst = train_pos_edge_index[1, idx]
            rel = train_pos_edge_attr[idx]
            pos_logits = model.score_logits(src, rel, dst)
            neg_losses: list[Tensor] = []
            g2 = torch.Generator(device=device)
            g2.manual_seed(int(cfg.seed) + epoch * 1_000_003 + batch_idx)
            for _ in range(cfg.neg_samples):
                dst_neg = _sample_tail_corruption(
                    src, dst, num_nodes=num_nodes, edge_set=edge_set, gen=g2
                )
                neg_logits = model.score_logits(src, rel, dst_neg)
                neg_losses.append(neg_logits)
            neg_stack = torch.stack(neg_losses, dim=1)
            logits = torch.cat([pos_logits.unsqueeze(1), neg_stack], dim=1)
            targets = torch.zeros_like(logits)
            targets[:, 0] = 1.0
            loss = F.binary_cross_entropy_with_logits(
                logits.reshape(-1), targets.reshape(-1), reduction="mean"
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            model.normalize_entities()

            epoch_loss += float(loss.item())
            n_batches += 1
            batch_idx += 1

        avg_loss = epoch_loss / max(1, n_batches)
        model.eval()
        with torch.no_grad():
            val_metrics = evaluate_rotate_link_prediction(
                model,
                pos_edge_index=val_pos_edge_index,
                neg_edge_index=val_neg_edge_index,
                negatives_per_pos=negatives_per_pos,
                query_uv_relation=query_uv_relation,
                device=device,
                batch_size=cfg.eval_batch_size,
                query_buckets=val_query_buckets,
                return_scores=False,
            )
        val_auc = float(val_metrics.get("roc_auc", float("nan")))
        val_ap = float(val_metrics.get("average_precision", float("nan")))
        pos_l, neg_l = _val_bce_components(
            model,
            val_pos_edge_index,
            val_neg_edge_index,
            negatives_per_pos,
            query_uv_relation,
            device,
            cfg.eval_batch_size,
        )
        val_loss = float(
            mean_bce_logits_link_prediction(
                pos_l, neg_l, negatives_per_pos=negatives_per_pos
            ).item()
        )

        history.epoch.append(epoch)
        history.train_loss.append(avg_loss)
        history.learning_rate.append(cfg.learning_rate)
        history.val_auc.append(val_auc)
        history.val_ap.append(val_ap)
        history.val_loss.append(val_loss)
        history.batch_count.append(n_batches)
        last_epoch = epoch

        cur_score = (
            val_auc
            if val_auc == val_auc
            else (val_ap if val_ap == val_ap else float("-inf"))
        )
        if cur_score > best_auc:
            best_auc = cur_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(cfg.early_stop_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                _LOG.info("RotatE early stop at epoch=%s best_selection_score=%s", epoch, best_auc)
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_val = evaluate_rotate_link_prediction(
            model,
            pos_edge_index=val_pos_edge_index,
            neg_edge_index=val_neg_edge_index,
            negatives_per_pos=negatives_per_pos,
            query_uv_relation=query_uv_relation,
            device=device,
            batch_size=cfg.eval_batch_size,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
        )
        final_test: dict | None = None
        if test_pos_edge_index is not None and test_neg_edge_index is not None:
            final_test = evaluate_rotate_link_prediction(
                model,
                pos_edge_index=test_pos_edge_index,
                neg_edge_index=test_neg_edge_index,
                negatives_per_pos=negatives_per_pos,
                query_uv_relation=query_uv_relation,
                device=device,
                batch_size=cfg.eval_batch_size,
                query_buckets=test_query_buckets,
                return_scores=return_scores,
            )

    return model, last_epoch, history, final_val, final_test


def _val_bce_components(
    model: RotatE,
    val_pos: Tensor,
    val_neg: Tensor,
    negatives_per_pos: int,
    query_uv_relation: dict[tuple[int, int], int],
    device: torch.device,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Flat pos logits and flat neg logits for BCE helper."""
    n_pos = int(val_pos.size(1))
    eli = torch.cat([val_pos, val_neg], dim=1).to(device)
    rel_ids = relation_ids_for_query_batch(
        val_pos.to(device),
        val_neg.to(device),
        query_uv_relation,
        negatives_per_pos,
        device=device,
    )
    src = eli[0]
    dst = eli[1]
    parts: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, int(src.numel()), batch_size):
            end = min(start + batch_size, int(src.numel()))
            parts.append(
                model.score_logits(
                    src[start:end], rel_ids[start:end], dst[start:end]
                )
            )
    scores = torch.cat(parts, dim=0)
    pos_logits = scores[:n_pos]
    neg_logits = scores[n_pos:].view(n_pos, negatives_per_pos)
    return pos_logits, neg_logits
