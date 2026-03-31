from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import networkx as nx
import torch

from kgml_new.config import TrainConfig
from kgml_new.runtime import resolve_device, seed_everything, validate_positive
from kgml_new.data.graph import networkx_to_data
from kgml_new.embeddings.semantic import (
    build_relation_tensor,
    relation_embeddings_from_graph,
)
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.eval import link_prediction_dot_product
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    train_unsupervised,
)
from kgml_new.training.splits import build_train_graph_data, create_edge_split


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
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. cuda, cuda:0, cpu",
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic CuDNN behavior for reproducibility",
    )
    p.add_argument(
        "--use-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mixed precision behavior",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="Save node embeddings .pt path"
    )
    args = p.parse_args()

    validate_positive("epochs", args.epochs)
    if args.batch_size is not None:
        validate_positive("batch_size", args.batch_size)
    seed_everything(args.seed, deterministic=args.deterministic)
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.use_amp is not None:
        cfg.use_amp = args.use_amp

    g = _load_graph(args.graph)
    data, relation_lookup = networkx_to_data(g, in_dim=cfg.in_dim, seed=cfg.seed)

    device = resolve_device(args.device)
    mask = data.edge_index[0] < data.edge_index[1]
    pos_edge_index = data.edge_index[:, mask].clone()
    pos_edge_attr = data.edge_attr[mask].clone() if hasattr(data, "edge_attr") else None
    split = create_edge_split(
        pos_edge_index,
        num_src_nodes=int(data.num_nodes),
        val_ratio=0.1,
        test_ratio=0.1,
        seed=cfg.seed,
        undirected=True,
    )
    train_pos = split.train_pos_edge_index
    train_pos_attr = None
    if pos_edge_attr is not None:
        key_to_idx = {
            (int(pos_edge_index[0, i]), int(pos_edge_index[1, i])): i
            for i in range(pos_edge_index.size(1))
        }
        train_indices = [
            key_to_idx[(int(train_pos[0, i]), int(train_pos[1, i]))]
            for i in range(train_pos.size(1))
        ]
        train_pos_attr = pos_edge_attr[torch.tensor(train_indices, dtype=torch.long)]
    train_data = build_train_graph_data(
        data, train_pos, train_pos_edge_attr=train_pos_attr
    )

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
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=False,
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=False)
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
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=True,
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=True)

    print(f"Embeddings shape: {tuple(z.shape)}")

    if split.val_pos_edge_index.numel() > 0:
        val_metrics = link_prediction_dot_product(
            z, split.val_pos_edge_index.to(z.device), split.val_neg_edge_index.to(z.device)
        )
        print("validation metrics:", val_metrics)
    if split.test_pos_edge_index.numel() > 0:
        test_metrics = link_prediction_dot_product(
            z, split.test_pos_edge_index.to(z.device), split.test_neg_edge_index.to(z.device)
        )
        print("test metrics:", test_metrics)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embeddings": z.cpu(), "relation_lookup": relation_lookup}, args.out
        )
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
