"""Relation ids for decode-time DistMult scoring (LinkNeighborLoader batches + eval queries)."""

from __future__ import annotations

import torch
from torch import Tensor


def relation_ids_for_uv_pairs(
    u: Tensor,
    v: Tensor,
    uv_rel: dict[tuple[int, int], int],
    *,
    device: torch.device,
) -> Tensor:
    """Map each pair (u_i, v_i) to a relation id via canonical undirected keys."""
    B = int(u.numel())
    out = torch.zeros(B, dtype=torch.long, device=device)
    if B == 0:
        return out
    uc = u.detach().cpu().long().tolist()
    vc = v.detach().cpu().long().tolist()
    for i in range(B):
        a, b = min(uc[i], vc[i]), max(uc[i], vc[i])
        out[i] = int(uv_rel.get((a, b), 0))
    return out


def build_canonical_uv_relation_lookup(
    edge_index: Tensor,
    edge_attr: Tensor | None,
) -> dict[tuple[int, int], int]:
    """Map undirected (min(u,v), max(u,v)) -> relation id from PyG edge list."""
    if edge_index.numel() == 0 or edge_attr is None:
        return {}
    ei = edge_index.cpu().long()
    ea = edge_attr.cpu().long()
    out: dict[tuple[int, int], int] = {}
    for i in range(ei.size(1)):
        u, v = int(ei[0, i]), int(ei[1, i])
        a, b = min(u, v), max(u, v)
        out[(a, b)] = int(ea[i].item())
    return out


def relation_ids_for_link_neighbor_batch(
    edge_label_index: Tensor,
    edge_label: Tensor,
    n_id: Tensor,
    train_lookup: dict[tuple[int, int], int],
    neg_samples: int,
    *,
    device: torch.device,
) -> Tensor:
    """
    Relation id per labeled edge for training.

    Positives: lookup global (u,v) from ``n_id`` and ``train_lookup``.
    Negatives: PyG LinkNeighborLoader appends ``neg_samples`` negatives per positive
    in order; each negative block inherits the corresponding positive relation id.
    """
    E = int(edge_label_index.size(1))
    n_pos = int((edge_label > 0.5).sum().item())
    out = torch.zeros(E, dtype=torch.long, device=device)
    if n_pos == 0:
        return out

    eli = edge_label_index
    n_id_d = n_id.to(device)
    g_src = n_id_d[eli[0, :n_pos]]
    g_dst = n_id_d[eli[1, :n_pos]]
    a = torch.minimum(g_src, g_dst)
    b = torch.maximum(g_src, g_dst)
    pos_rel = torch.zeros(n_pos, dtype=torch.long, device=device)
    for i in range(n_pos):
        pos_rel[i] = int(train_lookup.get((int(a[i].item()), int(b[i].item())), 0))
    out[:n_pos] = pos_rel

    rem = E - n_pos
    if rem <= 0:
        return out

    if neg_samples > 0 and rem == n_pos * neg_samples:
        out[n_pos:] = pos_rel.repeat_interleave(neg_samples)
    elif n_pos > 0 and rem % n_pos == 0:
        k = rem // n_pos
        out[n_pos:] = pos_rel.repeat_interleave(k)
    else:
        last = int(pos_rel[-1].item()) if n_pos > 0 else 0
        for i in range(n_pos, E):
            if edge_label[i] > 0.5:
                u = int(n_id_d[eli[0, i]].item())
                v = int(n_id_d[eli[1, i]].item())
                last = int(train_lookup.get((min(u, v), max(u, v)), 0))
            out[i] = last

    return out


def relation_ids_for_query_batch(
    pos_edge_index: Tensor,
    neg_edge_index: Tensor,
    uv_rel: dict[tuple[int, int], int],
    negatives_per_pos: int,
    *,
    device: torch.device,
) -> Tensor:
    """One relation id per column of [pos | neg_flat] matching evaluate_inductive layout."""
    pos = pos_edge_index.to(device)
    neg = neg_edge_index.to(device)
    n_pos = int(pos.size(1))
    E = n_pos + int(neg.size(1))
    out = torch.zeros(E, dtype=torch.long, device=device)
    for i in range(n_pos):
        u, v = int(pos[0, i]), int(pos[1, i])
        rid = int(uv_rel.get((min(u, v), max(u, v)), 0))
        out[i] = rid
        for k in range(negatives_per_pos):
            out[n_pos + i * negatives_per_pos + k] = rid
    return out
