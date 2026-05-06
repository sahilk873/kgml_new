from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn


class Node2VecEmbedding(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

    def forward(self) -> torch.Tensor:
        return self.embedding.weight


def _build_neighbors(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().detach().cpu().long().tolist():
        neighbors[int(src)].append(int(dst))
    return neighbors


def _sample_walk(
    *,
    start: int,
    neighbors: list[list[int]],
    walk_length: int,
    generator: torch.Generator,
) -> list[int]:
    walk = [int(start)]
    cur = int(start)
    for _ in range(max(walk_length - 1, 0)):
        nbrs = neighbors[cur]
        if not nbrs:
            break
        idx = int(torch.randint(0, len(nbrs), (1,), generator=generator).item())
        cur = int(nbrs[idx])
        walk.append(cur)
    return walk


def train_lightweight_node2vec(
    *,
    edge_index: torch.Tensor,
    num_nodes: int,
    embedding_dim: int,
    walk_length: int,
    context_size: int,
    walks_per_node: int,
    num_negative_samples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    use_amp: bool = False,
    grad_clip_norm: float = 1.0,
    checkpoint_path=None,
    resume: bool = False,
    save_every_epochs: int = 1,
    epoch_callback=None,
):
    """Pure-PyTorch Node2Vec-style fallback.

    This is intentionally simple: random walks define positive context pairs and
    in-batch negative sampling trains a skip-gram objective. It keeps the public
    API used by the training wrappers without requiring ``pyg-lib`` or
    ``torch-cluster``.
    """
    del checkpoint_path, resume, save_every_epochs
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    neighbors = _build_neighbors(edge_index, num_nodes)
    model = Node2VecEmbedding(num_nodes, embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Precompute a modest set of walks so training stays deterministic and cheap.
    walks: list[list[int]] = []
    for _ in range(max(walks_per_node, 1)):
        for node in range(num_nodes):
            walks.append(
                _sample_walk(
                    start=node,
                    neighbors=neighbors,
                    walk_length=walk_length,
                    generator=gen,
                )
            )

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and use_amp) else None
    last_epoch = -1
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        order = torch.randperm(len(walks), generator=gen).tolist()
        for start in range(0, len(order), max(batch_size, 1)):
            batch_walks = [walks[i] for i in order[start : start + batch_size]]
            if not batch_walks:
                continue
            centers = []
            contexts = []
            for walk in batch_walks:
                if len(walk) < 2:
                    continue
                for i, center in enumerate(walk):
                    left = max(0, i - context_size)
                    right = min(len(walk), i + context_size + 1)
                    for j in range(left, right):
                        if j == i:
                            continue
                        centers.append(center)
                        contexts.append(walk[j])
            if not centers:
                continue
            center_t = torch.tensor(centers, dtype=torch.long, device=device)
            context_t = torch.tensor(contexts, dtype=torch.long, device=device)
            neg_t = torch.randint(
                0,
                num_nodes,
                (context_t.numel() * max(num_negative_samples, 1),),
                generator=gen,
                device=device,
            )
            center_emb = model.embedding(center_t)
            context_emb = model.embedding(context_t)
            pos_logits = (center_emb * context_emb).sum(dim=-1)
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            neg_center = center_t.repeat_interleave(max(num_negative_samples, 1))
            neg_emb = model.embedding(neg_center)
            neg_logits = (neg_emb * model.embedding(neg_t)).sum(dim=-1)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )
            loss = pos_loss + neg_loss
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                opt.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        last_epoch = epoch
        if epoch_callback is not None:
            with torch.inference_mode():
                epoch_callback(epoch, total_loss / max(batches, 1), model)

    with torch.inference_mode():
        z = model().detach()
    return model, z, last_epoch
