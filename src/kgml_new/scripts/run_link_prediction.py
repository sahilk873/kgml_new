from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from kgml_new.config import (
    link_mlp_config_from_run_gpu_method_args,
    train_config_from_link_prediction_args,
)
from kgml_new.data.datasets import (
    compute_relation_diversity_buckets,
    prepare_link_prediction_dataset,
)
from kgml_new.data.loaders import load_pickled_graph
from kgml_new.embeddings.semantic import (
    build_relation_tensor,
    relation_embeddings_from_graph,
)
from kgml_new.models.baseline_sage import NEIGHBOR_AGGREGATIONS, BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EDGE_RELATION_MODES, build_edge_aware_model
from kgml_new.training.eval import evaluate_inductive_link_prediction
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    train_unsupervised,
)
from kgml_new.training.train_link_mlp import train_link_mlp_with_validation


def parse_args() -> argparse.Namespace:
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
    p.add_argument(
        "--edge-relation-mode",
        choices=EDGE_RELATION_MODES,
        default="concat",
        help="How projected relation embeddings control edge-aware message passing.",
    )
    p.add_argument(
        "--num-relation-bases",
        type=int,
        default=4,
        help="Number of shared basis message transforms for basis_mixture mode.",
    )
    p.add_argument(
        "--neighbor-aggr",
        choices=NEIGHBOR_AGGREGATIONS,
        default="mean",
        help="GraphSAGE neighbor aggregation: mean or max (pool).",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--train-batch-size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--edge-dim", type=int, default=None)
    p.add_argument("--num-neighbors-spec", type=str, default=None)
    p.add_argument("--link-mlp-learning-rate", type=float, default=None)
    p.add_argument("--link-mlp-dropout", type=float, default=None)
    p.add_argument("--link-mlp-batch-size", type=int, default=None)
    p.add_argument("--link-mlp-hidden-dims", type=str, default=None)
    p.add_argument(
        "--out", type=Path, default=None, help="Save node embeddings .pt path"
    )
    p.add_argument("--optuna-trials", type=int, default=0)
    p.add_argument(
        "--optuna-metric",
        choices=("val_auc", "val_ap"),
        default="val_auc",
    )
    p.add_argument("--optuna-storage", type=str, default=None)
    p.add_argument("--optuna-study", type=str, default=None)
    p.add_argument("--optuna-seed", type=int, default=None)
    p.add_argument("--optuna-timeout", type=float, default=None)
    return p.parse_args()


def run_link_prediction_experiment(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    cfg = train_config_from_link_prediction_args(args)

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
            neighbor_aggr=cfg.neighbor_aggr,
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
                embedding_model="openai",
            )
        else:
            rel_emb = {
                k: torch.zeros(cfg.edge_dim, dtype=torch.float32)
                for k in relation_lookup
            }

        rel_table = build_relation_tensor(
            rel_emb, relation_lookup, cfg.edge_dim, device
        )
        model = build_edge_aware_model(
            edge_relation_mode=args.edge_relation_mode,
            in_channels=cfg.in_dim,
            edge_dim=cfg.edge_dim,
            hidden_channels=cfg.hidden_dim,
            out_channels=cfg.out_dim,
            relation_table=rel_table,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            concat=cfg.concat,
            normalize_output=True,
            num_relation_bases=args.num_relation_bases,
            neighbor_aggr=cfg.neighbor_aggr,
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
            config=link_mlp_config_from_run_gpu_method_args(args),
            device=device,
        )

    val_metrics = None
    test_metrics = None
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

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embeddings": z.cpu(), "relation_lookup": relation_lookup}, args.out
        )

    return {
        "split_protocol": args.split_protocol,
        "negative_sampling_mode": dataset.negative_sampling_mode,
        "decoder": args.decoder,
        "edge_relation_mode": args.edge_relation_mode,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "embedding_shape": list(z.shape),
        "saved_embeddings": str(args.out) if args.out else None,
    }


def main() -> None:
    args = parse_args()

    if args.optuna_trials > 0:
        from kgml_new.tuning.optuna_runner import require_optuna
        from kgml_new.tuning.spaces import (
            apply_param_dict_to_namespace,
            suggest_link_prediction_legacy_params,
        )

        optuna = require_optuna()
        optuna_seed = int(args.optuna_seed if args.optuna_seed is not None else args.seed)
        sampler = optuna.samplers.TPESampler(seed=optuna_seed)
        if args.optuna_storage:
            study = optuna.create_study(
                study_name=args.optuna_study or f"kgml_link_pred_{args.model}",
                storage=args.optuna_storage,
                direction="maximize",
                sampler=sampler,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial: optuna.Trial) -> float:
            t_args = copy.deepcopy(args)
            suggest_link_prediction_legacy_params(trial, t_args)
            torch.manual_seed(optuna_seed + trial.number * 100_003)
            out = run_link_prediction_experiment(t_args)
            vm = out.get("val_metrics")
            if not vm:
                return float("-inf")
            key = "roc_auc" if args.optuna_metric == "val_auc" else "average_precision"
            v = float(vm.get(key) or 0.0)
            return v if v == v else float("-inf")

        study.optimize(
            _objective,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
        )
        t_args = copy.deepcopy(args)
        apply_param_dict_to_namespace(t_args, study.best_params)
        torch.manual_seed(args.seed)
        summary = run_link_prediction_experiment(t_args)
        summary["optuna"] = {
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "metric": args.optuna_metric,
        }
        print(json.dumps(summary, indent=2, default=str))
        return

    out = run_link_prediction_experiment(args)
    print(f"split protocol: {args.split_protocol}")
    print(f"negative sampling mode: {out['negative_sampling_mode']}")
    print(f"decoder: {args.decoder}")
    print(f"edge relation mode: {args.edge_relation_mode}")
    print(f"Embeddings shape: {tuple(out['embedding_shape'])}")
    if out.get("val_metrics"):
        print("validation metrics:", out["val_metrics"])
    if out.get("test_metrics"):
        print("test metrics:", out["test_metrics"])
    if out.get("saved_embeddings"):
        print(f"Saved {out['saved_embeddings']}")


if __name__ == "__main__":
    main()
