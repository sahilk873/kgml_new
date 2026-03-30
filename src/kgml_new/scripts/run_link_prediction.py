from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import networkx as nx
import torch

from kgml_new.config import TrainConfig
from kgml_new.data.graph import networkx_to_data
from kgml_new.embeddings.semantic import (
    build_relation_tensor,
    relation_embeddings_from_graph,
)
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.eval import link_prediction_sklearn
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    train_unsupervised,
)


def _load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, nx.Graph):
        return obj
    raise TypeError(f"Expected networkx.Graph, got {type(obj)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Unsupervised GraphSAGE / edge-aware semantic GraphSAGE link prediction"
    )
    p.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Pickle file containing a networkx.Graph",
    )
    p.add_argument(
        "--model",
        choices=("sage", "edge_sage"),
        default="sage",
        help="sage: SAGEConv baseline; edge_sage: relation-aware semantic",
    )
    p.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For edge_sage: use OpenAI embeddings (requires kgml-new[semantic] and OPENAI_API_KEY)",
    )
    p.add_argument(
        "--cache", type=Path, default=None, help="Pickle cache for relation embeddings"
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out", type=Path, default=None, help="Save node embeddings .pt path"
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed)

    g = _load_graph(args.graph)
    data, relation_lookup = networkx_to_data(g, in_dim=cfg.in_dim, seed=cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = data.edge_index[0] < data.edge_index[1]
    train_pos = data.edge_index[:, mask].clone()

    if args.model == "sage":
        model = BaselineGraphSAGE(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.out_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        train_unsupervised(
            model,
            data,
            train_pos,
            cfg,
            device=device,
            edge_aware=False,
        )
        z = compute_node_embeddings(model, data, device, edge_aware=False)
    else:
        if args.semantic:
            rel_emb = relation_embeddings_from_graph(
                g,
                edge_dim=cfg.edge_dim,
                cache_path=args.cache,
                use_openai=True,
            )
        else:
            rel_emb = {
                k: torch.zeros(cfg.edge_dim, dtype=torch.float32)
                for k in relation_lookup
            }

        rel_table = build_relation_tensor(
            rel_emb, relation_lookup, cfg.edge_dim, device
        )
        model = EdgeAwareGraphSAGE(
            cfg.in_dim,
            cfg.edge_dim,
            cfg.hidden_dim,
            cfg.out_dim,
            relation_table=rel_table,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            concat=cfg.concat,
        )
        train_unsupervised(
            model,
            data,
            train_pos,
            cfg,
            device=device,
            edge_aware=True,
        )
        z = compute_node_embeddings(model, data, device, edge_aware=True)

    print(f"Embeddings shape: {tuple(z.shape)}")

    n_edges = data.edge_index.size(1) // 2
    if n_edges > 1:
        mid = n_edges // 2
        ei = data.edge_index[:, :n_edges]
        pos = ei[:, :mid]
        neg = torch.randint(0, data.num_nodes, (2, mid), device=z.device)
        metrics = link_prediction_sklearn(z.cpu(), pos.cpu(), neg.cpu())
        print("quick eval (random negatives):", metrics)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embeddings": z.cpu(), "relation_lookup": relation_lookup}, args.out
        )
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
