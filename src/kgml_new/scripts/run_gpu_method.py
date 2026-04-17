from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import torch

from kgml_new.config import (
    TrainConfig,
    link_mlp_config_from_run_gpu_method_args,
    node2vec_config_from_run_gpu_method_args,
    train_config_from_run_gpu_method_args,
)
from kgml_new.logging_config import parse_log_level, setup_kgml_logging
from kgml_new.data.datasets import (
    compute_relation_diversity_buckets,
    prepare_link_prediction_dataset,
)
from kgml_new.data.loaders import (
    GraphCSVSpec,
    PRIMEKG_CSV_SPEC,
    load_graph_csv,
    load_pickled_graph,
)
from kgml_new.data.prepared_dataset_cache import (
    graph_csv_spec_to_dict,
    input_fingerprint,
    load_prepared_link_prediction_dataset,
)
from kgml_new.embeddings.semantic import (
    DEFAULT_GLOSSARY_PATH,
    build_relation_tensor,
    compute_semantic_similarity_matrix,
    relation_embeddings_from_graph,
)
from kgml_new.models.baseline_gcn import BaselineGCN
from kgml_new.models.baseline_sage import NEIGHBOR_AGGREGATIONS, BaselineGraphSAGE
from kgml_new.models.edge_aware_sage import EDGE_RELATION_MODES, build_edge_aware_model
from kgml_new.models.semantic_relation_inject import (
    FilmSageSemanticInjectBundle,
    RelDistMultDecodeHead,
)
from kgml_new.training.relation_decode_batch import build_canonical_uv_relation_lookup
from kgml_new.eval.ood_difficulty import (
    OODDifficultyConfig,
    build_ood_json_payload,
    compute_ood_difficulty_for_split,
    strip_scores_from_metrics,
    write_edge_predictions_csv,
)
from kgml_new.training.eval import (
    compute_relation_holdout_metrics,
    evaluate_inductive_link_prediction,
    grouped_link_prediction_metrics,
)
from kgml_new.training.link_unsupervised import (
    compute_node_embeddings,
    train_unsupervised_batched,
    train_unsupervised_fullgraph,
)
from kgml_new.training.node2vec_train import (
    train_node2vec_embeddings_with_validation,
)
from kgml_new.training.train_link_mlp import train_link_mlp_with_validation
from kgml_new.runtime import resolve_device, seed_everything

_LOG = logging.getLogger("kgml_new.scripts.run_gpu_method")


def _grouped_dot_metrics(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    negatives_per_pos: int,
) -> dict[str, float]:
    pos_scores = (
        (z[pos_edge_index[0].long()] * z[pos_edge_index[1].long()]).sum(dim=-1).detach().cpu().numpy()
    )
    neg_scores_flat = (
        (z[neg_edge_index[0].long()] * z[neg_edge_index[1].long()]).sum(dim=-1).detach().cpu()
    )
    if neg_scores_flat.numel() % negatives_per_pos != 0:
        raise ValueError(
            "neg_edge_index count must be divisible by negatives_per_pos "
            f"({neg_scores_flat.numel()} vs {negatives_per_pos})"
        )
    neg_scores = neg_scores_flat.view(-1, negatives_per_pos).numpy()
    return grouped_link_prediction_metrics(pos_scores, neg_scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one KGML method on a generic KG and save JSON results."
    )
    parser.add_argument(
        "--method",
        choices=(
            "baseline_sage",
            "baseline_gcn",
            "edge_aware_sage",
            "edge_aware_sage_node_emb",
            "edge_aware_sage_film_semdec",
            "node2vec",
            "link_mlp",
            "edge_aware_link_mlp",
        ),
        required=True,
    )
    parser.add_argument(
        "--input", "--csv", dest="input_path", type=Path, default=Path("data/kg.csv")
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "pickle"),
        default="auto",
    )
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--source-col", type=str, default=PRIMEKG_CSV_SPEC.source_col)
    parser.add_argument("--target-col", type=str, default=PRIMEKG_CSV_SPEC.target_col)
    parser.add_argument(
        "--relation-col", type=str, default=PRIMEKG_CSV_SPEC.relation_col
    )
    parser.add_argument(
        "--source-type-col", type=str, default=PRIMEKG_CSV_SPEC.source_type_col
    )
    parser.add_argument(
        "--target-type-col", type=str, default=PRIMEKG_CSV_SPEC.target_type_col
    )
    parser.add_argument(
        "--split-protocol",
        choices=("node", "node_category", "edge"),
        default="node",
        help="node: random node-disjoint split; node_category: hold out node types (see "
        "--held-out-node-categories); edge: edge-disjoint split.",
    )
    parser.add_argument(
        "--negative-sampling-mode",
        choices=("global", "type_matched"),
        default=None,
    )
    parser.add_argument("--negatives-per-pos", type=int, default=20)
    parser.add_argument("--decoder", choices=("dot", "mlp"), default="dot")
    parser.add_argument(
        "--shuffle-relations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--held-out-relations",
        nargs="*",
        default=[],
        metavar="REL",
        help="Relation type names held out from training positives (zero-shot at val/test). "
        "Repeat flag or pass multiple names.",
    )
    parser.add_argument(
        "--held-out-node-categories",
        nargs="*",
        default=[],
        metavar="TYPE",
        help="With --split-protocol node_category: node_type values (graph node attribute) "
        "held out from training positives; val/test query edges touch partitioned held-out nodes.",
    )
    parser.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For edge-aware methods: use semantic relation embeddings by default.",
    )
    parser.add_argument(
        "--semantic-cache",
        type=Path,
        default=None,
        help="Optional cache path for relation embeddings used by edge-aware methods.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=["openai", "gemini", "sapbert", "e5", "random"],
        default=None,
        help="Embedding model for relation embeddings: openai, gemini, sapbert (PubMedBERT), e5, or random.",
    )
    parser.add_argument(
        "--sapbert-model",
        type=str,
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
        help="HuggingFace model name for SapBERT",
    )
    parser.add_argument(
        "--e5-model",
        type=str,
        default="intfloat/e5-base-v2",
        help="HuggingFace model name for E5",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model name for embeddings.",
    )
    parser.add_argument(
        "--strict-semantic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of falling back if embedding setup fails.",
    )
    parser.add_argument(
        "--glossary-path",
        type=Path,
        default=None,
        help="Path to relation glossary TSV file.",
    )
    parser.add_argument(
        "--relation-text-mode",
        type=str,
        choices=("raw", "canonical"),
        default="raw",
        help="Text mode for relation prompts and semantic caches (raw vs canonical glossary rewrite).",
    )
    parser.add_argument(
        "--edge-relation-mode",
        choices=EDGE_RELATION_MODES,
        default="concat",
        help="How projected relation embeddings control edge-aware message passing.",
    )
    parser.add_argument(
        "--concat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only for edge_relation_mode=concat: concatenate relation embedding with neighbor "
        "features before the message linear. Gated, basis_mixture, and film always use "
        "relation features in their own message rules; this flag does not change them.",
    )
    parser.add_argument(
        "--num-relation-bases",
        type=int,
        default=4,
        help="Number of shared basis message transforms for basis_mixture mode.",
    )
    parser.add_argument(
        "--neighbor-aggr",
        choices=NEIGHBOR_AGGREGATIONS,
        default="mean",
        help="GraphSAGE neighbor aggregation: mean (default) or max (Hamilton et al. pool).",
    )
    parser.add_argument(
        "--semantic-alignment-lambda",
        type=float,
        default=0.0,
        help="Weight for semantic alignment regularizer in basis_mixture mode.",
    )
    parser.add_argument(
        "--rel-residual-scale",
        type=float,
        default=0.25,
        help="For edge_aware_sage_film_semdec: scale of trainable MLP residual on frozen "
        "relation rows (encoder) before DistMult-style decode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (matches TrainConfig default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-dim", type=int, default=64)
    parser.add_argument(
        "--node-embeddings-path",
        type=Path,
        default=None,
        help="Optional .pt file with node embeddings [num_nodes, dim]. Required for "
        "edge_aware_sage_node_emb.",
    )
    parser.add_argument(
        "--node-embeddings-key",
        type=str,
        default=None,
        help="If --node-embeddings-path stores a dict, pick this key. If omitted, tries "
        "common keys: embeddings, node_embeddings, x, features.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device, e.g. cuda, cuda:0, cpu. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit if CUDA is not available (use on GPU Slurm jobs to fail fast).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append structured run logs to this file (stderr always receives logs too).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level for kgml_new loggers.",
    )
    parser.add_argument(
        "--prepared-dataset-cache",
        type=Path,
        default=None,
        help="Load PyG tensors + split + negatives from export_prepared_link_prediction.py. "
        "Pass --input as the same graph file used to build the cache (metadata fingerprint).",
    )
    parser.add_argument(
        "--compute-ood-difficulty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Post-hoc OOD difficulty stratified metrics (requires evaluator scores).",
    )
    parser.add_argument(
        "--save-edge-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write per-edge CSV under --ood-output-dir (use with --compute-ood-difficulty).",
    )
    parser.add_argument(
        "--ood-buckets",
        choices=("quantile", "fixed"),
        default="quantile",
        help="Bucket mode: quantile (default); fixed is not implemented and falls back to quantile.",
    )
    parser.add_argument(
        "--ood-num-quantile-buckets",
        type=int,
        default=3,
        help="Number of quantile buckets (e.g. 3 for tertiles).",
    )
    parser.add_argument(
        "--ood-tail-quantile",
        type=float,
        default=0.1,
        help="Tail mass for nodefreq_tail / nodefreq_head extremes.",
    )
    parser.add_argument(
        "--ood-eval-splits",
        nargs="+",
        choices=("val", "test"),
        default=("val", "test"),
        help="Which splits to include in OOD JSON and CSV.",
    )
    parser.add_argument(
        "--ood-output-dir",
        type=Path,
        default=Path("results/ood"),
        help="Directory for OOD CSV exports.",
    )
    parser.add_argument(
        "--ood-run-name",
        type=str,
        default=None,
        help="Stem for OOD CSV filename (default: --output stem).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam LR for unsupervised encoder training (default: TrainConfig).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Mini-batch size for batched unsupervised training (default: TrainConfig).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout on GNN layers (default: TrainConfig).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of message-passing layers (default: TrainConfig).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden channel width (default: TrainConfig).",
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=None,
        help="Relation embedding width for edge-aware paths (default: TrainConfig).",
    )
    parser.add_argument(
        "--num-neighbors-spec",
        type=str,
        default=None,
        help="Comma-separated neighbor sample sizes per layer, e.g. 10,10.",
    )
    parser.add_argument(
        "--link-mlp-learning-rate",
        type=float,
        default=None,
        help="Adam LR for link MLP decoder head (default: LinkMLPConfig).",
    )
    parser.add_argument(
        "--link-mlp-dropout",
        type=float,
        default=None,
        help="Dropout for link MLP (default: LinkMLPConfig).",
    )
    parser.add_argument(
        "--link-mlp-batch-size",
        type=int,
        default=None,
        help="Batch size for link MLP (default: LinkMLPConfig).",
    )
    parser.add_argument(
        "--link-mlp-hidden-dims",
        type=str,
        default=None,
        help='Hidden dims for link MLP, comma-separated, e.g. "256,128".',
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="If >0, run Optuna hyperparameter search (requires pip install optuna).",
    )
    parser.add_argument(
        "--optuna-metric",
        choices=("val_auc", "val_ap"),
        default="val_auc",
        help="Validation metric to maximize during Optuna.",
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help="Optuna RDB storage URL, e.g. sqlite:///study.db",
    )
    parser.add_argument(
        "--optuna-study",
        type=str,
        default=None,
        help="Study name when using --optuna-storage.",
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=None,
        help="Sampler seed for Optuna (default: --seed).",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=float,
        default=None,
        help="Optional wall-clock timeout in seconds for the whole study.",
    )
    return parser.parse_args()


def _csv_spec_from_args(args: argparse.Namespace) -> GraphCSVSpec:
    return GraphCSVSpec(
        source_col=args.source_col,
        target_col=args.target_col,
        relation_col=args.relation_col,
        source_type_col=args.source_type_col,
        target_type_col=args.target_type_col,
        source_id_col=PRIMEKG_CSV_SPEC.source_id_col,
        target_id_col=PRIMEKG_CSV_SPEC.target_id_col,
        node_type_attr=PRIMEKG_CSV_SPEC.node_type_attr,
        relation_attr=PRIMEKG_CSV_SPEC.relation_attr,
        default_node_type=PRIMEKG_CSV_SPEC.default_node_type,
        edge_attr_cols=PRIMEKG_CSV_SPEC.edge_attr_cols,
        directed=False,
    )


def _load_graph(args: argparse.Namespace):
    input_format = args.input_format
    if input_format == "auto":
        suffix = args.input_path.suffix.lower()
        input_format = "pickle" if suffix in {".pkl", ".pickle"} else "csv"
    if input_format == "pickle":
        return load_pickled_graph(args.input_path)
    return load_graph_csv(
        args.input_path, spec=_csv_spec_from_args(args), max_edges=args.max_edges
    )


def _held_out_relations_from_args(args: argparse.Namespace) -> list[str] | None:
    names = [str(x).strip() for x in getattr(args, "held_out_relations", []) or []]
    names = [x for x in names if x]
    return names or None


def _held_out_node_categories_from_args(args: argparse.Namespace) -> list[str] | None:
    names = [str(x).strip() for x in getattr(args, "held_out_node_categories", []) or []]
    names = [x for x in names if x]
    return names or None


def build_dataset(args: argparse.Namespace):
    graph = _load_graph(args)
    dataset = prepare_link_prediction_dataset(
        graph,
        in_dim=args.in_dim,
        seed=args.seed,
        val_ratio=0.1,
        test_ratio=0.1,
        split_protocol=args.split_protocol,
        negative_sampling_mode=args.negative_sampling_mode,
        negatives_per_pos=args.negatives_per_pos,
        decoder=args.decoder,
        shuffle_relations=args.shuffle_relations,
        held_out_relations=_held_out_relations_from_args(args),
        held_out_node_categories=_held_out_node_categories_from_args(args),
    )
    return graph, dataset


def _resolved_input_format_for_args(args: argparse.Namespace) -> str:
    if args.input_format != "auto":
        return str(args.input_format)
    suffix = args.input_path.suffix.lower()
    return "pickle" if suffix in {".pkl", ".pickle"} else "csv"


def _expected_prepared_cache_meta(args: argparse.Namespace) -> dict:
    rf = _resolved_input_format_for_args(args)
    neg_resolved = args.negative_sampling_mode
    if neg_resolved is None:
        neg_resolved = (
            "type_matched"
            if args.split_protocol in ("node", "node_category")
            else "global"
        )
    spec_dict = (
        graph_csv_spec_to_dict(_csv_spec_from_args(args)) if rf == "csv" else None
    )
    ho = _held_out_relations_from_args(args)
    ho_cat = _held_out_node_categories_from_args(args)
    meta = {
        "split_protocol": args.split_protocol,
        "seed": args.seed,
        "in_dim": args.in_dim,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "negatives_per_pos": args.negatives_per_pos,
        "decoder": args.decoder,
        "shuffle_relations": args.shuffle_relations,
        "add_self_loops": False,
        "max_edges": args.max_edges,
        "negative_sampling_mode_resolved": neg_resolved,
        "negative_sampling_mode_cli": args.negative_sampling_mode,
        "resolved_input_format": rf,
        "graph_csv_spec": spec_dict,
        "input": input_fingerprint(Path(args.input_path).resolve()),
    }
    if ho:
        meta["held_out_relations"] = sorted(ho)
    if ho_cat:
        meta["held_out_node_categories"] = sorted(ho_cat)
    return meta


def history_dict(history) -> dict:
    return {
        "epoch": history.epoch,
        "train_loss": history.train_loss,
        "val_auc": history.val_auc,
        "val_ap": history.val_ap,
        "batch_count": history.batch_count,
    }


def _resolved_embedding_model(args: argparse.Namespace) -> str:
    if args.embedding_model is not None:
        return str(args.embedding_model)
    return "random" if not args.semantic else "openai"


def _extract_node_embedding_tensor(
    payload: object,
    *,
    key: str | None,
) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        if key is not None:
            value = payload.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Node embedding key '{key}' not found as a tensor in payload."
                )
            return value
        for candidate in ("embeddings", "node_embeddings", "x", "features"):
            value = payload.get(candidate)
            if isinstance(value, torch.Tensor):
                return value
        raise ValueError(
            "Node embedding payload is a dict but contains no tensor under "
            "keys [embeddings, node_embeddings, x, features]. "
            "Use --node-embeddings-key to select the correct tensor."
        )
    raise ValueError(
        "Unsupported node embedding payload type. Expected torch.Tensor or dict."
    )


def _maybe_apply_external_node_embeddings(
    *,
    args: argparse.Namespace,
    dataset,
    required: bool,
) -> torch.Tensor | None:
    emb_path = args.node_embeddings_path
    if emb_path is None:
        if required:
            raise SystemExit(
                "This method requires --node-embeddings-path pointing to a .pt tensor "
                "with shape [num_nodes, dim]."
            )
        return None
    if not emb_path.is_file():
        raise SystemExit(f"Node embeddings file not found: {emb_path}")

    payload = torch.load(emb_path, map_location="cpu")
    node_emb = _extract_node_embedding_tensor(payload, key=args.node_embeddings_key)
    if node_emb.ndim != 2:
        raise SystemExit(
            f"Node embeddings must be rank-2 [num_nodes, dim], got shape={tuple(node_emb.shape)}"
        )
    if node_emb.size(0) != int(dataset.train_data.num_nodes):
        raise SystemExit(
            "Node embedding row count mismatch: "
            f"embeddings have {int(node_emb.size(0))} rows but graph has "
            f"{int(dataset.train_data.num_nodes)} nodes."
        )
    if not torch.is_floating_point(node_emb):
        node_emb = node_emb.float()
    node_emb = node_emb.detach().cpu().to(torch.float32)

    dataset.data.x = node_emb.clone()
    dataset.train_data.x = node_emb.clone()
    dataset.full_data.x = node_emb.clone()
    return node_emb


def _maybe_build_semantic_similarity_for_alignment(
    *,
    args: argparse.Namespace,
    graph,
    relation_lookup: dict[str, int],
    relation_table: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    if (
        args.semantic_alignment_lambda <= 0
        or args.edge_relation_mode != "basis_mixture"
        or not args.semantic
    ):
        return None
    emb_model = _resolved_embedding_model(args)
    glossary_path = args.glossary_path or DEFAULT_GLOSSARY_PATH
    rel_emb = relation_embeddings_from_graph(
        graph,
        edge_dim=int(relation_table.size(-1)),
        cache_path=args.semantic_cache,
        embedding_model=emb_model,
        relation_text_mode=args.relation_text_mode,
        strict_embedding=args.strict_semantic,
        glossary_path=glossary_path,
        sapbert_model=args.sapbert_model,
        e5_model=args.e5_model,
        gemini_model=args.gemini_model,
    )
    semantic_sim_matrix = compute_semantic_similarity_matrix(
        rel_emb, relation_lookup
    ).to(device)
    print(
        f"Semantic alignment enabled: lambda={args.semantic_alignment_lambda}, "
        f"sim_matrix shape={tuple(semantic_sim_matrix.shape)}"
    )
    return semantic_sim_matrix


def build_edge_aware_relation_table(
    *,
    args: argparse.Namespace,
    training_graph,
    relation_lookup: dict[str, int],
    edge_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    embedding_model = _resolved_embedding_model(args)
    glossary_path = args.glossary_path or DEFAULT_GLOSSARY_PATH

    rel_emb = relation_embeddings_from_graph(
        training_graph,
        edge_dim=edge_dim,
        cache_path=args.semantic_cache,
        embedding_model=embedding_model,
        relation_text_mode=args.relation_text_mode,
        strict_embedding=args.strict_semantic,
        glossary_path=glossary_path,
        sapbert_model=args.sapbert_model,
        e5_model=args.e5_model,
        gemini_model=args.gemini_model,
    )
    relation_table = build_relation_tensor(rel_emb, relation_lookup, edge_dim, device)
    relation_init = (
        f"{embedding_model}_semantic" if embedding_model != "random" else "random"
    )
    return relation_table, relation_init


def build_edge_aware_encoder(
    *,
    cfg: TrainConfig,
    relation_table: torch.Tensor,
    normalize_output: bool = True,
    film_semantic_inject: bool = False,
    rel_residual_scale: float = 0.25,
):
    return build_edge_aware_model(
        edge_relation_mode=cfg.edge_relation_mode,
        in_channels=cfg.in_dim,
        edge_dim=cfg.edge_dim,
        hidden_channels=cfg.hidden_dim,
        out_channels=cfg.out_dim,
        relation_table=relation_table,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        concat=cfg.concat,
        normalize_output=normalize_output,
        num_relation_bases=cfg.num_relation_bases,
        neighbor_aggr=cfg.neighbor_aggr,
        film_semantic_inject=film_semantic_inject,
        rel_residual_scale=rel_residual_scale,
    )


def maybe_train_decoder(
    *,
    decoder: str,
    z: torch.Tensor,
    train_pos: torch.Tensor,
    epochs: int,
    device: torch.device,
    run_args: argparse.Namespace,
):
    if decoder == "dot":
        return None, None, None
    mlp_cfg = link_mlp_config_from_run_gpu_method_args(run_args, epochs=epochs)
    decoder_model, decoder_last_epoch, decoder_history = train_link_mlp_with_validation(
        z.detach(),
        train_pos,
        mlp_cfg,
        device=device,
    )
    return decoder_model, decoder_last_epoch, decoder_history


def _attach_relation_holdout_block(
    output: dict,
    *,
    dataset,
    val_metrics: dict | None,
    test_metrics: dict | None,
    val_pos: torch.Tensor,
    test_pos: torch.Tensor,
) -> None:
    hid = getattr(dataset, "held_out_relation_ids", None)
    if not hid or dataset.positive_edge_attr is None:
        return
    if val_metrics is None or test_metrics is None:
        return
    if "pos_scores" not in val_metrics or "pos_scores" not in test_metrics:
        _LOG.warning(
            "Relation holdout metrics skipped: run with scores enabled "
            "(use --compute-ood-difficulty or held-out relations trigger scores automatically)."
        )
        return
    pos_idx = dataset.positive_edge_index
    pos_attr = dataset.positive_edge_attr
    output["relation_holdout"] = {
        "held_out_relations": sorted(dataset.held_out_relations)
        if dataset.held_out_relations
        else [],
        "held_out_relation_ids": sorted(hid),
        "val": compute_relation_holdout_metrics(
            query_pos_edge_index=val_pos,
            positive_edge_index=pos_idx,
            positive_edge_attr=pos_attr,
            held_out_relation_ids=hid,
            pos_scores=val_metrics["pos_scores"],
            neg_scores=val_metrics["neg_scores"],
        ),
        "test": compute_relation_holdout_metrics(
            query_pos_edge_index=test_pos,
            positive_edge_index=pos_idx,
            positive_edge_attr=pos_attr,
            held_out_relation_ids=hid,
            pos_scores=test_metrics["pos_scores"],
            neg_scores=test_metrics["neg_scores"],
        ),
    }


def _finalize_eval_and_ood(
    output: dict,
    *,
    val_metrics: dict | None,
    test_metrics: dict | None,
    args: argparse.Namespace,
    dataset,
    train_pos: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
) -> None:
    _attach_relation_holdout_block(
        output,
        dataset=dataset,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_pos=val_pos,
        test_pos=test_pos,
    )
    if val_metrics is not None:
        output["val_metrics"] = strip_scores_from_metrics(val_metrics)
    if test_metrics is not None:
        output["test_metrics"] = strip_scores_from_metrics(test_metrics)
        output["metrics"] = output["test_metrics"]

    if not args.compute_ood_difficulty:
        return
    if val_metrics is None or test_metrics is None:
        _LOG.warning("OOD requested but val/test metrics missing; skipping.")
        return
    if "pos_scores" not in val_metrics or "pos_scores" not in test_metrics:
        _LOG.warning("OOD requested but evaluator did not return scores; skipping.")
        return

    if args.ood_buckets == "fixed":
        _LOG.warning("--ood-buckets=fixed not implemented; using quantile buckets.")

    ood_cfg = OODDifficultyConfig(
        num_quantile_buckets=args.ood_num_quantile_buckets,
        tail_quantile=args.ood_tail_quantile,
        bucket_mode="quantile",
    )
    train_attr = dataset.train_pos_edge_attr
    num_nodes = int(dataset.train_data.num_nodes)
    splits_requested = set(args.ood_eval_splits)

    val_block = None
    test_block = None
    if "val" in splits_requested:
        val_block = compute_ood_difficulty_for_split(
            split_name="val",
            train_pos_edge_index=train_pos,
            train_pos_edge_attr=train_attr,
            full_edge_index=dataset.full_data.edge_index,
            positive_edge_index=dataset.positive_edge_index,
            positive_edge_attr=dataset.positive_edge_attr,
            query_pos_edge_index=val_pos,
            pos_scores=val_metrics["pos_scores"],
            neg_scores=val_metrics["neg_scores"],
            negatives_per_pos=dataset.negatives_per_pos,
            num_nodes=num_nodes,
            relation_lookup=dataset.relation_lookup,
            split_protocol=args.split_protocol,
            config=ood_cfg,
        )
    if "test" in splits_requested:
        test_block = compute_ood_difficulty_for_split(
            split_name="test",
            train_pos_edge_index=train_pos,
            train_pos_edge_attr=train_attr,
            full_edge_index=dataset.full_data.edge_index,
            positive_edge_index=dataset.positive_edge_index,
            positive_edge_attr=dataset.positive_edge_attr,
            query_pos_edge_index=test_pos,
            pos_scores=test_metrics["pos_scores"],
            neg_scores=test_metrics["neg_scores"],
            negatives_per_pos=dataset.negatives_per_pos,
            num_nodes=num_nodes,
            relation_lookup=dataset.relation_lookup,
            split_protocol=args.split_protocol,
            config=ood_cfg,
        )

    payload = build_ood_json_payload(val_block=val_block, test_block=test_block)
    payload["config"] = {
        "ood_buckets": args.ood_buckets,
        "ood_num_quantile_buckets": args.ood_num_quantile_buckets,
        "ood_tail_quantile": args.ood_tail_quantile,
        "ood_eval_splits": list(args.ood_eval_splits),
    }
    output["ood_difficulty"] = payload

    if not args.save_edge_predictions:
        return

    ood_stem = args.ood_run_name or args.output.stem
    csv_path = Path(args.ood_output_dir) / f"{ood_stem}_edge_predictions.csv"
    edge_mode = (
        args.edge_relation_mode
        if args.method
        in (
            "edge_aware_sage",
            "edge_aware_sage_node_emb",
            "edge_aware_sage_film_semdec",
            "edge_aware_link_mlp",
        )
        else None
    )
    wrote_any = False
    if val_block is not None and val_block.get("features") is not None:
        write_edge_predictions_csv(
            csv_path,
            split_name="val",
            features=val_block["features"],
            pos_scores=val_metrics["pos_scores"],
            neg_scores=val_metrics["neg_scores"],
            query_neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            relation_lookup=dataset.relation_lookup,
            seed=args.seed,
            model_name=args.method,
            edge_relation_mode=edge_mode,
            split_protocol=args.split_protocol,
            append=False,
        )
        wrote_any = True
    if test_block is not None and test_block.get("features") is not None:
        write_edge_predictions_csv(
            csv_path,
            split_name="test",
            features=test_block["features"],
            pos_scores=test_metrics["pos_scores"],
            neg_scores=test_metrics["neg_scores"],
            query_neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            relation_lookup=dataset.relation_lookup,
            seed=args.seed,
            model_name=args.method,
            edge_relation_mode=edge_mode,
            split_protocol=args.split_protocol,
            append=wrote_any,
        )
    _LOG.info("phase=ood_csv path=%s", csv_path.resolve())


def run_gpu_method_experiment(
    args: argparse.Namespace,
    *,
    device: torch.device,
    graph,
    dataset,
) -> dict:
    return_scores = args.compute_ood_difficulty or bool(
        getattr(dataset, "held_out_relation_ids", None)
    )
    train_data = dataset.train_data
    relation_lookup = dataset.relation_lookup
    split = dataset.split
    train_pos = split.train_pos_edge_index
    val_pos = split.val_pos_edge_index
    val_neg = split.val_neg_edge_index
    test_pos = split.test_pos_edge_index
    test_neg = split.test_neg_edge_index
    diversity_counts, diversity_buckets = compute_relation_diversity_buckets(
        dataset.graph,
        dataset.node_list,
    )
    val_query_buckets = [diversity_buckets[int(node)] for node in val_pos[0].tolist()]
    test_query_buckets = [diversity_buckets[int(node)] for node in test_pos[0].tolist()]

    output = {
        "method": args.method,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "num_nodes": int(train_data.num_nodes),
        "num_train_edges": int(train_pos.size(1)),
        "num_val_edges": int(val_pos.size(1)),
        "num_test_edges": int(test_pos.size(1)),
        "input_path": str(args.input_path),
        "input_format": args.input_format,
        "split_protocol": args.split_protocol,
        "negative_sampling_mode": dataset.negative_sampling_mode,
        "negatives_per_pos": dataset.negatives_per_pos,
        "decoder": args.decoder,
        "shuffle_relations": args.shuffle_relations,
        "semantic": args.semantic,
        "semantic_cache": str(args.semantic_cache) if args.semantic_cache else None,
        "strict_semantic": args.strict_semantic,
        "embedding_model": args.embedding_model,
        "embedding_model_resolved": _resolved_embedding_model(args),
        "relation_text_mode": args.relation_text_mode,
        "glossary_path": str(args.glossary_path)
        if args.glossary_path is not None
        else str(DEFAULT_GLOSSARY_PATH),
        "sapbert_model": args.sapbert_model,
        "e5_model": args.e5_model,
        "gemini_model": args.gemini_model,
        "semantic_alignment_lambda": args.semantic_alignment_lambda,
        "edge_relation_mode": args.edge_relation_mode,
        "num_relation_bases": args.num_relation_bases,
        "max_edges": args.max_edges,
        "in_dim": args.in_dim,
        "epochs": args.epochs,
        "evaluation_protocol": (
            "sampled inductive eval on full adjacency with scored edges removed"
            if args.split_protocol in ("node", "node_category")
            else "sampled eval on full adjacency with scored edges removed"
        ),
        "relation_diversity_summary": {
            "min": min(diversity_counts.values()) if diversity_counts else 0,
            "max": max(diversity_counts.values()) if diversity_counts else 0,
        },
        "edge_message_concat": args.concat
        if args.method
        in (
            "edge_aware_sage",
            "edge_aware_sage_node_emb",
            "edge_aware_sage_film_semdec",
            "edge_aware_link_mlp",
        )
        and args.edge_relation_mode == "concat"
        else None,
        "neighbor_aggr": args.neighbor_aggr
        if args.method
        in (
            "baseline_sage",
            "edge_aware_sage",
            "edge_aware_sage_node_emb",
            "edge_aware_sage_film_semdec",
            "link_mlp",
            "edge_aware_link_mlp",
        )
        else None,
        "held_out_relations": sorted(dataset.held_out_relations)
        if getattr(dataset, "held_out_relations", None)
        else [],
        "held_out_relation_ids": sorted(dataset.held_out_relation_ids)
        if getattr(dataset, "held_out_relation_ids", None)
        else [],
        "held_out_node_categories": sorted(dataset.held_out_node_categories)
        if getattr(dataset, "held_out_node_categories", None)
        else [],
        "node_embeddings_path": str(args.node_embeddings_path)
        if args.node_embeddings_path is not None
        else None,
        "node_embeddings_key": args.node_embeddings_key,
    }

    if args.method == "baseline_sage":
        _LOG.info(
            "phase=model baseline_sage neighbor_aggr=%s edge_mode=n/a",
            args.neighbor_aggr,
        )
        cfg = train_config_from_run_gpu_method_args(args)
        model = BaselineGraphSAGE(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.out_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            neighbor_aggr=cfg.neighbor_aggr,
        )
        _LOG.info("phase=train_start (baseline_sage batched)")
        model, last_epoch, history = train_unsupervised_batched(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=False,
        )
        _LOG.info("phase=train_done last_epoch=%s", last_epoch)
        z = compute_node_embeddings(model, train_data, device, edge_aware=False)
        decoder_model, decoder_last_epoch, decoder_history = maybe_train_decoder(
            decoder=args.decoder,
            z=z,
            train_pos=train_pos,
            epochs=args.epochs,
            device=device,
            run_args=args,
        )
        _LOG.info("phase=eval_val (baseline_sage)")
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
        )
        _LOG.info("phase=eval_test (baseline_sage)")
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
            return_scores=return_scores,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if decoder_history is not None:
            output["decoder_last_epoch"] = decoder_last_epoch
            output["decoder_history"] = history_dict(decoder_history)
        _finalize_eval_and_ood(
            output,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            args=args,
            dataset=dataset,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )

    elif args.method == "baseline_gcn":
        cfg = train_config_from_run_gpu_method_args(args)
        model = BaselineGCN(
            cfg.in_dim,
            cfg.hidden_dim,
            cfg.out_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        model, last_epoch, history = train_unsupervised_fullgraph(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=False,
        )
        z = compute_node_embeddings(model, train_data, device, edge_aware=False)
        decoder_model, decoder_last_epoch, decoder_history = maybe_train_decoder(
            decoder=args.decoder,
            z=z,
            train_pos=train_pos,
            epochs=args.epochs,
            device=device,
            run_args=args,
        )
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
        )
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
            return_scores=return_scores,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if decoder_history is not None:
            output["decoder_last_epoch"] = decoder_last_epoch
            output["decoder_history"] = history_dict(decoder_history)
        _finalize_eval_and_ood(
            output,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            args=args,
            dataset=dataset,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )

    elif args.method == "edge_aware_sage":
        _LOG.info(
            "phase=model edge_aware_sage neighbor_aggr=%s edge_relation_mode=%s "
            "semantic_cache=%s embedding_resolved=%s",
            args.neighbor_aggr,
            args.edge_relation_mode,
            args.semantic_cache,
            _resolved_embedding_model(args),
        )
        cfg = train_config_from_run_gpu_method_args(args)
        edge_dim = cfg.edge_dim
        _LOG.info("phase=relation_table (edge_aware)")
        relation_table, relation_init = build_edge_aware_relation_table(
            args=args,
            training_graph=dataset.graph,
            relation_lookup=relation_lookup,
            edge_dim=edge_dim,
            device=device,
        )
        model = build_edge_aware_encoder(cfg=cfg, relation_table=relation_table)

        semantic_sim_matrix = _maybe_build_semantic_similarity_for_alignment(
            args=args,
            graph=dataset.graph,
            relation_lookup=relation_lookup,
            relation_table=relation_table,
            device=device,
        )

        _LOG.info("phase=train_start (edge_aware_sage batched)")
        model, last_epoch, history = train_unsupervised_batched(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=True,
            semantic_similarity_matrix=semantic_sim_matrix,
            semantic_alignment_lambda=args.semantic_alignment_lambda,
            edge_attr_for_alignment=train_data.edge_attr,
            alignment_train_pos_edge_index=train_pos,
            alignment_train_pos_edge_attr=dataset.train_pos_edge_attr,
        )
        _LOG.info("phase=train_done last_epoch=%s", last_epoch)
        z = compute_node_embeddings(model, train_data, device, edge_aware=True)
        decoder_model, decoder_last_epoch, decoder_history = maybe_train_decoder(
            decoder=args.decoder,
            z=z,
            train_pos=train_pos,
            epochs=args.epochs,
            device=device,
            run_args=args,
        )
        _LOG.info("phase=eval_val (edge_aware_sage)")
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
        )
        _LOG.info("phase=eval_test (edge_aware_sage)")
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
            return_scores=return_scores,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["relation_init"] = relation_init
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if decoder_history is not None:
            output["decoder_last_epoch"] = decoder_last_epoch
            output["decoder_history"] = history_dict(decoder_history)
        _finalize_eval_and_ood(
            output,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            args=args,
            dataset=dataset,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )

    elif args.method == "edge_aware_sage_node_emb":
        _LOG.info(
            "phase=model edge_aware_sage_node_emb neighbor_aggr=%s edge_relation_mode=%s "
            "semantic_cache=%s embedding_resolved=%s",
            args.neighbor_aggr,
            args.edge_relation_mode,
            args.semantic_cache,
            _resolved_embedding_model(args),
        )
        external_x = _maybe_apply_external_node_embeddings(
            args=args,
            dataset=dataset,
            required=True,
        )
        if external_x is None:
            raise RuntimeError("Expected external node embeddings for edge_aware_sage_node_emb.")
        cfg = train_config_from_run_gpu_method_args(args)
        node_dim = int(external_x.size(1))
        if cfg.in_dim != node_dim or cfg.out_dim != node_dim:
            _LOG.info(
                "phase=node_emb_dim_override old_in_dim=%s old_out_dim=%s new_dim=%s",
                cfg.in_dim,
                cfg.out_dim,
                node_dim,
            )
            cfg = replace(cfg, in_dim=node_dim, out_dim=node_dim)
        edge_dim = cfg.edge_dim
        _LOG.info("phase=relation_table (edge_aware_node_emb)")
        relation_table, relation_init = build_edge_aware_relation_table(
            args=args,
            training_graph=dataset.graph,
            relation_lookup=relation_lookup,
            edge_dim=edge_dim,
            device=device,
        )
        model = build_edge_aware_encoder(cfg=cfg, relation_table=relation_table)

        semantic_sim_matrix = _maybe_build_semantic_similarity_for_alignment(
            args=args,
            graph=dataset.graph,
            relation_lookup=relation_lookup,
            relation_table=relation_table,
            device=device,
        )

        _LOG.info("phase=train_start (edge_aware_sage_node_emb batched)")
        model, last_epoch, history = train_unsupervised_batched(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=True,
            semantic_similarity_matrix=semantic_sim_matrix,
            semantic_alignment_lambda=args.semantic_alignment_lambda,
            edge_attr_for_alignment=train_data.edge_attr,
            alignment_train_pos_edge_index=train_pos,
            alignment_train_pos_edge_attr=dataset.train_pos_edge_attr,
        )
        _LOG.info("phase=train_done last_epoch=%s", last_epoch)
        z = compute_node_embeddings(model, train_data, device, edge_aware=True)
        decoder_model, decoder_last_epoch, decoder_history = maybe_train_decoder(
            decoder=args.decoder,
            z=z,
            train_pos=train_pos,
            epochs=args.epochs,
            device=device,
            run_args=args,
        )
        _LOG.info("phase=eval_val (edge_aware_sage_node_emb)")
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
        )
        _LOG.info("phase=eval_test (edge_aware_sage_node_emb)")
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder=args.decoder,
            decoder_model=decoder_model,
            query_buckets=test_query_buckets,
            return_scores=return_scores,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["relation_init"] = relation_init
        output["node_embedding_dim"] = node_dim
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if decoder_history is not None:
            output["decoder_last_epoch"] = decoder_last_epoch
            output["decoder_history"] = history_dict(decoder_history)
        _finalize_eval_and_ood(
            output,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            args=args,
            dataset=dataset,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )

    elif args.method == "edge_aware_sage_film_semdec":
        if args.edge_relation_mode != "film":
            _LOG.warning(
                "edge_aware_sage_film_semdec uses FiLM + semantic adapter; forcing "
                "edge_relation_mode from %s to film.",
                args.edge_relation_mode,
            )
            args.edge_relation_mode = "film"
        if args.decoder != "dot":
            _LOG.warning(
                "edge_aware_sage_film_semdec trains a relation-conditioned DistMult decode "
                "head; ignoring --decoder=%s for eval (dot + rel_decode_bundle).",
                args.decoder,
            )
        _LOG.info(
            "phase=model edge_aware_sage_film_semdec neighbor_aggr=%s rel_residual_scale=%s "
            "semantic_cache=%s embedding_resolved=%s",
            args.neighbor_aggr,
            args.rel_residual_scale,
            args.semantic_cache,
            _resolved_embedding_model(args),
        )
        cfg = train_config_from_run_gpu_method_args(args)
        edge_dim = cfg.edge_dim
        _LOG.info("phase=relation_table (edge_aware_sage_film_semdec)")
        relation_table, relation_init = build_edge_aware_relation_table(
            args=args,
            training_graph=dataset.graph,
            relation_lookup=relation_lookup,
            edge_dim=edge_dim,
            device=device,
        )
        encoder = build_edge_aware_encoder(
            cfg=cfg,
            relation_table=relation_table,
            film_semantic_inject=True,
            rel_residual_scale=float(args.rel_residual_scale),
        )
        rel_dim = int(relation_table.shape[1])
        decode_head = RelDistMultDecodeHead(rel_dim, cfg.out_dim)
        model = FilmSageSemanticInjectBundle(encoder, decode_head)

        semantic_sim_matrix = _maybe_build_semantic_similarity_for_alignment(
            args=args,
            graph=dataset.graph,
            relation_lookup=relation_lookup,
            relation_table=relation_table,
            device=device,
        )

        train_uv_relation = build_canonical_uv_relation_lookup(
            train_pos.cpu(),
            dataset.train_pos_edge_attr.cpu(),
        )

        _LOG.info("phase=train_start (edge_aware_sage_film_semdec batched)")
        model, last_epoch, history = train_unsupervised_batched(
            model,
            train_data,
            train_pos,
            cfg,
            device=device,
            edge_aware=True,
            semantic_similarity_matrix=semantic_sim_matrix,
            semantic_alignment_lambda=args.semantic_alignment_lambda,
            edge_attr_for_alignment=train_data.edge_attr,
            alignment_train_pos_edge_index=train_pos,
            alignment_train_pos_edge_attr=dataset.train_pos_edge_attr,
            relation_decode_bundle=model,
            train_uv_relation_lookup=train_uv_relation,
            metric_eval_data=dataset.full_data,
        )
        _LOG.info("phase=train_done last_epoch=%s", last_epoch)
        z = compute_node_embeddings(model, train_data, device, edge_aware=True)
        _LOG.info("phase=eval_val (edge_aware_sage_film_semdec)")
        val_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder="dot",
            decoder_model=None,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
            rel_decode_bundle=model,
        )
        _LOG.info("phase=eval_test (edge_aware_sage_film_semdec)")
        test_metrics = evaluate_inductive_link_prediction(
            model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=cfg.num_neighbors,
            decoder="dot",
            decoder_model=None,
            query_buckets=test_query_buckets,
            return_scores=return_scores,
            rel_decode_bundle=model,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["relation_init"] = relation_init
        output["semantic_inject"] = "film_adapter_distmult"
        output["rel_residual_scale"] = float(args.rel_residual_scale)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        _finalize_eval_and_ood(
            output,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            args=args,
            dataset=dataset,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )

    elif args.method == "node2vec":
        cfg = node2vec_config_from_run_gpu_method_args(args)
        _, z, last_epoch, history = train_node2vec_embeddings_with_validation(
            train_data,
            cfg,
            device=device,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        if args.split_protocol in ("node", "node_category"):
            output["warning"] = (
                "node2vec was trained on the train-node subgraph only; unseen nodes do not "
                "receive inductive embeddings, so this run is excluded from the main node-split comparison."
            )
            output["comparable_under_node_split"] = False
        else:
            output["val_metrics"] = _grouped_dot_metrics(
                z, val_pos, val_neg, args.negatives_per_pos
            )
            output["test_metrics"] = _grouped_dot_metrics(
                z, test_pos, test_neg, args.negatives_per_pos
            )
            output["metrics"] = output["test_metrics"]
            output["comparable_under_node_split"] = True
        if args.compute_ood_difficulty:
            _LOG.warning(
                "OOD difficulty is not available for node2vec (no per-edge score export); "
                "use baseline_sage, edge_aware_sage, edge_aware_sage_node_emb, "
                "edge_aware_sage_film_semdec, link_mlp, or edge_aware_link_mlp."
            )

    elif args.method == "link_mlp":
        base_cfg = train_config_from_run_gpu_method_args(args)
        base_model = BaselineGraphSAGE(
            base_cfg.in_dim,
            base_cfg.hidden_dim,
            base_cfg.out_dim,
            num_layers=base_cfg.num_layers,
            dropout=base_cfg.dropout,
            neighbor_aggr=base_cfg.neighbor_aggr,
        )
        base_model, base_last_epoch, base_history = train_unsupervised_batched(
            base_model,
            train_data,
            train_pos,
            base_cfg,
            device=device,
            edge_aware=False,
        )
        z = compute_node_embeddings(
            base_model, train_data, device, edge_aware=False
        ).detach()
        mlp_cfg = link_mlp_config_from_run_gpu_method_args(args)
        mlp, last_epoch, history = train_link_mlp_with_validation(
            z,
            train_pos,
            mlp_cfg,
            device=device,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["base_embedding_history"] = history_dict(base_history)
        output["base_last_epoch"] = base_last_epoch
        val_metrics = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
        )
        test_metrics = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=False,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=test_query_buckets,
            return_scores=return_scores,
        )
        _finalize_eval_and_ood(
            output,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            args=args,
            dataset=dataset,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )

    else:
        base_cfg = train_config_from_run_gpu_method_args(args)
        edge_dim = base_cfg.edge_dim
        relation_table, relation_init = build_edge_aware_relation_table(
            args=args,
            training_graph=dataset.graph,
            relation_lookup=relation_lookup,
            edge_dim=edge_dim,
            device=device,
        )
        base_model = build_edge_aware_encoder(
            cfg=base_cfg, relation_table=relation_table
        )
        semantic_sim_matrix = _maybe_build_semantic_similarity_for_alignment(
            args=args,
            graph=dataset.graph,
            relation_lookup=relation_lookup,
            relation_table=relation_table,
            device=device,
        )
        base_model, base_last_epoch, base_history = train_unsupervised_batched(
            base_model,
            train_data,
            train_pos,
            base_cfg,
            device=device,
            edge_aware=True,
            semantic_similarity_matrix=semantic_sim_matrix,
            semantic_alignment_lambda=args.semantic_alignment_lambda,
            edge_attr_for_alignment=train_data.edge_attr,
            alignment_train_pos_edge_index=train_pos,
            alignment_train_pos_edge_attr=dataset.train_pos_edge_attr,
        )
        z = compute_node_embeddings(
            base_model, train_data, device, edge_aware=True
        ).detach()
        mlp_cfg = link_mlp_config_from_run_gpu_method_args(args)
        mlp, last_epoch, history = train_link_mlp_with_validation(
            z,
            train_pos,
            mlp_cfg,
            device=device,
        )
        output["train_embedding_shape"] = list(z.shape)
        output["relation_init"] = relation_init
        output["last_epoch"] = last_epoch
        output["history"] = history_dict(history)
        output["base_embedding_history"] = history_dict(base_history)
        output["base_last_epoch"] = base_last_epoch
        val_metrics = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=val_pos,
            neg_edge_index=val_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=val_query_buckets,
            return_scores=return_scores,
        )
        test_metrics = evaluate_inductive_link_prediction(
            base_model,
            dataset.full_data,
            pos_edge_index=test_pos,
            neg_edge_index=test_neg,
            negatives_per_pos=dataset.negatives_per_pos,
            device=device,
            edge_aware=True,
            num_neighbors=base_cfg.num_neighbors,
            decoder="mlp",
            decoder_model=mlp,
            query_buckets=test_query_buckets,
            return_scores=return_scores,
        )
        _finalize_eval_and_ood(
            output,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            args=args,
            dataset=dataset,
            train_pos=train_pos,
            val_pos=val_pos,
            val_neg=val_neg,
            test_pos=test_pos,
            test_neg=test_neg,
        )

    return output

def main() -> None:
    args = parse_args()
    setup_kgml_logging(log_file=args.log_file, level=parse_log_level(args.log_level))
    _LOG.info(
        "run_gpu_method start argv=%s cwd=%s",
        sys.argv,
        Path.cwd(),
    )
    seed_everything(args.seed)

    if args.require_cuda and not torch.cuda.is_available():
        _LOG.error("CUDA required but torch.cuda.is_available() is False")
        raise SystemExit(
            "CUDA required (--require-cuda) but unavailable. "
            "Use a GPU node, load drivers, and install a CUDA PyTorch wheel."
        )
    device = resolve_device(args.device)
    _LOG.info(
        "device=%s cuda_available=%s gpu_name=%s",
        device,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    _LOG.info(
        "phase=data method=%s input=%s max_edges=%s split=%s epochs=%s",
        args.method,
        args.input_path,
        args.max_edges,
        args.split_protocol,
        args.epochs,
    )
    if args.prepared_dataset_cache is not None:
        pcache = Path(args.prepared_dataset_cache)
        if not pcache.is_file():
            raise SystemExit(f"Prepared dataset cache not found: {pcache}")
        _LOG.info("phase=data source=prepared_dataset_cache path=%s", pcache)
        dataset, _ = load_prepared_link_prediction_dataset(
            pcache,
            expected_meta=_expected_prepared_cache_meta(args),
        )
        graph = dataset.graph
    else:
        graph, dataset = build_dataset(args)
    _LOG.info(
        "phase=data_done num_train_nodes=%s train_edges=%s",
        int(dataset.train_data.num_nodes),
        int(dataset.split.train_pos_edge_index.size(1)),
    )
    if args.optuna_trials > 0:
        if args.method == "node2vec" and args.split_protocol in (
            "node",
            "node_category",
        ):
            raise SystemExit(
                "Optuna (--optuna-trials > 0) is not supported for node2vec with "
                "--split-protocol node or node_category (no validation metrics). "
                "Use --split-protocol edge or disable Optuna."
            )
        from kgml_new.tuning.optuna_runner import require_optuna
        from kgml_new.tuning.spaces import (
            apply_param_dict_to_namespace,
            suggest_run_gpu_method_params,
        )

        optuna = require_optuna()
        optuna_seed = int(args.optuna_seed if args.optuna_seed is not None else args.seed)
        sampler = optuna.samplers.TPESampler(seed=optuna_seed)
        if args.optuna_storage:
            study = optuna.create_study(
                study_name=args.optuna_study or f"kgml_gpu_{args.method}",
                storage=args.optuna_storage,
                direction="maximize",
                sampler=sampler,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial: optuna.Trial) -> float:
            t_args = copy.deepcopy(args)
            suggest_run_gpu_method_params(trial, t_args)
            seed_everything(optuna_seed + trial.number * 100_003)
            out = run_gpu_method_experiment(
                t_args, device=device, graph=graph, dataset=dataset
            )
            vm = out.get("val_metrics")
            if not vm:
                return float("-inf")
            key = "roc_auc" if args.optuna_metric == "val_auc" else "average_precision"
            v = float(vm.get(key) or 0.0)
            if v != v:
                return float("-inf")
            tm = out.get("test_metrics") or {}
            trial.set_user_attr("test_roc_auc", tm.get("roc_auc"))
            trial.set_user_attr("test_average_precision", tm.get("average_precision"))
            return v

        study.optimize(
            _objective,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
        )
        t_args = copy.deepcopy(args)
        apply_param_dict_to_namespace(t_args, study.best_params)
        seed_everything(optuna_seed)
        output = run_gpu_method_experiment(
            t_args, device=device, graph=graph, dataset=dataset
        )
        output["optuna"] = {
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_params,
            "metric": args.optuna_metric,
            "storage": args.optuna_storage,
            "study_name": getattr(study, "study_name", None),
        }
    else:
        output = run_gpu_method_experiment(
            args, device=device, graph=graph, dataset=dataset
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _LOG.info("phase=write_json path=%s", args.output.resolve())
    args.output.write_text(json.dumps(output, indent=2))
    _LOG.info(
        "run_gpu_method done method=%s test_roc_auc=%s",
        args.method,
        output.get("test_metrics", {}).get("roc_auc"),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
