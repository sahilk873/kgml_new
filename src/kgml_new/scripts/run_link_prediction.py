from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kgml_new.config import LinkMLPConfig, TrainConfig
from kgml_new.data.datasets import (
    compute_relation_diversity_buckets,
    prepare_link_prediction_dataset,
)
from kgml_new.data.loaders import load_pickled_graph
from kgml_new.embeddings.semantic import (
    build_relation_tensor,
    relation_embeddings_from_graph,
)
from kgml_new.models.baseline_sage import BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EdgeAwareGraphSAGE
from kgml_new.training.eval import evaluate_inductive_link_prediction
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    train_unsupervised,
)
from kgml_new.training.train_link_mlp import train_link_mlp_with_validation


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
        "--split-protocol",
        choices=("node", "edge"),
        default="node",
        help="node: hold out entities from training edges for OOD evaluation; edge: legacy edge-disjoint split",
    )
    p.add_argument(
        "--negative-sampling-mode",
        choices=("global", "type_matched"),
        default=None,
    )
    p.add_argument("--negatives-per-pos", type=int, default=20)
    p.add_argument("--decoder", choices=("dot", "mlp"), default="dot")
    p.add_argument(
        "--shuffle-relations",
        action=argparse.BooleanOptionalAction,
        default=False,
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

    g = load_pickled_graph(args.graph)
    dataset = prepare_link_prediction_dataset(
        g,
        in_dim=cfg.in_dim,
        seed=cfg.seed,
        val_ratio=0.1,
        test_ratio=0.1,
        split_protocol=args.split_protocol,
        negative_sampling_mode=args.negative_sampling_mode,
        negatives_per_pos=args.negatives_per_pos,
        decoder=args.decoder,
        shuffle_relations=args.shuffle_relations,
    )
    print(f"split protocol: {args.split_protocol}")
    print(f"negative sampling mode: {dataset.negative_sampling_mode}")
    print(f"decoder: {args.decoder}")
    print(f"shuffle relations: {dataset.shuffle_relations}")
    relation_lookup = dataset.relation_lookup
    split = dataset.split
    train_pos = split.train_pos_edge_index
    train_data = dataset.train_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, diversity_buckets = compute_relation_diversity_buckets(dataset.graph, dataset.node_list)
    val_query_buckets = [diversity_buckets[int(node)] for node in split.val_pos_edge_index[0].tolist()]
    test_query_buckets = [diversity_buckets[int(node)] for node in split.test_pos_edge_index[0].tolist()]
    decoder_model = None

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
                dataset.graph,
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

    if args.decoder == "mlp":
        decoder_model, _, _ = train_link_mlp_with_validation(
            z.detach(),
            train_pos,
            config=LinkMLPConfig(
                epochs=args.epochs,
                seed=args.seed,
                feature_mode="concat_product",
            ),
            device=device,
        )

    print(f"Embeddings shape: {tuple(z.shape)}")

    if split.val_pos_edge_index.numel() > 0:
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=split.val_pos_edge_index,
            neg_edge_index=split.val_neg_edge_index,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=args.model == "edge_sage",
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
        )
        print("validation metrics:", val_metrics)
    if split.test_pos_edge_index.numel() > 0:
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=split.test_pos_edge_index,
            neg_edge_index=split.test_neg_edge_index,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=args.model == "edge_sage",
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
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
