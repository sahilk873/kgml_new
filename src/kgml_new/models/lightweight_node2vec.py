from __future__ import annotations

import random
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Node2VecEmbedding(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        sparse: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples

        self.embedding = nn.Embedding(num_nodes, embedding_dim, sparse=sparse)
        self.embedding.weight.data.uniform_(-1, 1)

    def forward(self, batch: Tensor) -> Tensor:
        return self.embedding(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        src_pos = self.embedding(pos_rw[:, 0])
        dst_pos = self.embedding(pos_rw[:, 1])
        pos_score = (src_pos * dst_pos).sum(dim=-1)
        pos_loss = -F.logsigmoid(pos_score).mean()

        src_neg = self.embedding(neg_rw[:, 0])
        dst_neg = self.embedding(neg_rw[:, 1])
        neg_score = (src_neg * dst_neg).sum(dim=-1)
        neg_loss = -F.logsigmoid(-neg_score).mean()

        return pos_loss + neg_loss


def train_lightweight_node2vec(
    edge_index: Tensor,
    num_nodes: int,
    embedding_dim: int = 64,
    walk_length: int = 20,
    context_size: int = 10,
    walks_per_node: int = 5,
    num_negative_samples: int = 1,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    checkpoint_path=None,
    resume=False,
    save_every_epochs: int = 1,
    epoch_callback=None,
):
    rng = random.Random(seed)

    row, col = edge_index.cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for i, j in zip(row, col):
        adj[i].append(j)
        adj[j].append(i)

    neighbors = [set(neighbors) for neighbors in adj]
    non_neighbors = [
        [j for j in range(num_nodes) if j != i and j not in neighbors[i]]
        for i in range(num_nodes)
    ]

    model = Node2VecEmbedding(
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=1.0,
        q=1.0,
        num_negative_samples=num_negative_samples,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def random_walk(start: int) -> list[int]:
        walk = [start]
        curr = start
        for _ in range(walk_length - 1):
            neighbors = adj[curr]
            if not neighbors:
                break
            curr = rng.choice(neighbors)
            walk.append(curr)
        return walk

    def generate_walks() -> Iterator[tuple[list[int], list[int]]]:
        for _ in range(walks_per_node):
            for node in range(num_nodes):
                walk = random_walk(node)
                for i in range(len(walk) - context_size + 1):
                    context = walk[i : i + context_size]
                    yield walk[i], context

    last_epoch = -1
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and use_amp) else None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0

        pos_rw, neg_rw = [], []
        for src, ctx in generate_walks():
            for dst in ctx[1:]:
                pos_rw.append([src, dst])
                for _ in range(num_negative_samples):
                    neg_node = rng.choice(non_neighbors[src])
                    neg_rw.append([src, neg_node])

            if len(pos_rw) >= batch_size:
                pos_rw_t = torch.tensor(pos_rw, dtype=torch.long, device=device)
                neg_rw_t = torch.tensor(neg_rw, dtype=torch.long, device=device)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
                    loss = model.loss(pos_rw_t, neg_rw_t)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

                total_loss += float(loss.detach())
                n += 1
                pos_rw, neg_rw = [], []
                
                del pos_rw_t, neg_rw_t, loss

        if pos_rw:
            pos_rw_t = torch.tensor(pos_rw, dtype=torch.long, device=device)
            neg_rw_t = torch.tensor(neg_rw, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=scaler is not None):
                loss = model.loss(pos_rw_t, neg_rw_t)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            total_loss += float(loss.detach())
            n += 1
            
            del pos_rw_t, neg_rw_t, loss

        last_epoch = epoch
        avg = total_loss / max(n, 1)
        if epoch_callback is not None:
            epoch_callback(epoch, avg, model)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"node2vec(light) epoch {epoch:04d} loss={avg:.4f}")

    model.eval()
    with torch.inference_mode():
        z = model.embedding.weight.data

    return model, z, last_epoch
