#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from kgml_new.config import Node2VecConfig
from kgml_new.data.graph import networkx_to_data
from kgml_new.training.node2vec_train import train_node2vec_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train node2vec on PrimeKG kg.csv and save UMAP cache.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--json-output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--walk-length", type=int, default=20)
    p.add_argument("--context-size", type=int, default=15)
    p.add_argument("--walks-per-node", type=int, default=4)
    p.add_argument("--p", type=float, default=1.0)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--num-negative-samples", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.01)
    return p.parse_args()


def load_primekg_edges(path: Path) -> tuple[torch.Tensor, list[str], list[str], list[str]]:
    usecols = ["x_index", "y_index", "x_name", "y_name", "x_type", "y_type"]
    edges: list[tuple[int, int]] = []
    names: dict[int, str] = {}
    types: dict[int, str] = {}
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        xs = chunk["x_index"].astype("int64")
        ys = chunk["y_index"].astype("int64")
        edges.extend(zip(xs.tolist(), ys.tolist(), strict=False))
        for idx, name, typ in zip(xs, chunk["x_name"].astype(str), chunk["x_type"].astype(str), strict=False):
            names.setdefault(int(idx), name)
            types.setdefault(int(idx), typ)
        for idx, name, typ in zip(ys, chunk["y_name"].astype(str), chunk["y_type"].astype(str), strict=False):
            names.setdefault(int(idx), name)
            types.setdefault(int(idx), typ)
    if not edges:
        raise ValueError(f"No edges found in {path}")
    num_nodes = max(max(u, v) for u, v in edges) + 1
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    node_names = [names.get(i, str(i)) for i in range(num_nodes)]
    node_types = [types.get(i, "unknown") for i in range(num_nodes)]
    node_ids = [str(i) for i in range(num_nodes)]
    return edge_index, node_names, node_types, node_ids


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    torch.manual_seed(args.seed)
    edge_index, node_names, node_types, node_ids = load_primekg_edges(args.input)
    num_nodes = len(node_names)
    device = torch.device(args.device)

    import networkx as nx

    g = nx.Graph()
    for i in range(num_nodes):
        g.add_node(i, node_type=node_types[i], name=node_names[i], node_id=node_ids[i])
    for src, dst in edge_index.t().tolist():
        g.add_edge(int(src), int(dst), relationship="kg")
    data, _ = networkx_to_data(g, in_dim=args.embed_dim, seed=args.seed)
    cfg = Node2VecConfig(
        embedding_dim=args.embed_dim,
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        p=args.p,
        q=args.q,
        num_negative_samples=args.num_negative_samples,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    model, embeddings, _ = train_node2vec_embeddings(data, cfg, device=device)
    losses = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "node_types": node_types,
            "node_names": node_names,
            "node_ids": node_ids,
            "source": str(args.input),
        },
        args.output,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_nodes": num_nodes,
            "embedding_dim": args.embed_dim,
            "config": vars(args),
        },
        args.model_output,
    )
    args.json_output.write_text(
        json.dumps(
            {
                "method": "node2vec",
                "source": str(args.input),
                "num_nodes": num_nodes,
                "embedding_shape": list(embeddings.shape),
                "artifacts": {
                    "train_embeddings": str(args.output),
                    "model_state_dict": str(args.model_output),
                },
                "history": {"train_loss": losses},
                "config": {
                    **{
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    "input": str(args.input),
                    "output": str(args.output),
                },
            },
            indent=2,
        )
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.model_output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
