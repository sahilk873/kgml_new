from __future__ import annotations

import random
from collections import deque
from typing import Iterator

import torch
import torch.nn as nn
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
    checkpoint_path=None,
    resume=False,
    save_every_epochs: int = 1,
):
    import numpy as np

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    row, col = edge_index.cpu().numpy()
    adj = [[] for _ in range(num_nodes)]
    for i, j in zip(row, col):
        adj[i].append(j)
        adj[j].append(i)

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
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0

        pos_rw, neg_rw = [], []
        for src, ctx in generate_walks():
            pos_rw.append([src] * len(ctx))
            pos_rw.append(ctx)
            for _ in range(num_negative_samples):
                neg_node = rng.randint(0, num_nodes - 1)
                neg_rw.append([src, neg_node])

            if len(pos_rw) >= batch_size:
                pos_rw_t = torch.tensor(pos_rw, dtype=torch.long, device=device)
                neg_rw_t = torch.tensor(neg_rw, dtype=torch.long, device=device)

                optimizer.zero_grad()
                loss = model.loss(pos_rw_t, neg_rw_t)
                loss.backward()
                optimizer.step()

                total_loss += float(loss)
                n += 1
                pos_rw, neg_rw = [], []

        last_epoch = epoch
        if epoch % 20 == 0 or epoch == epochs - 1:
            avg = total_loss / max(n, 1)
            print(f"node2vec(light) epoch {epoch:04d} loss={avg:.4f}")

    model.eval()
    with torch.inference_mode():
        z = model.embedding.weight.weight.data

    return model, z, last_epoch
