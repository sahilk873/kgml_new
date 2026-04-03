from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kgml_new.data.loaders import (
    GraphCSVSpec,
    PRIMEKG_CSV_SPEC,
    _has_header,
    _infer_delimiter,
    load_pickled_graph,
)
from kgml_new.data.relations import get_edge_types
from kgml_new.embeddings.semantic import (
    DEFAULT_GLOSSARY_PATH,
    describe_relation,
    relation_embeddings_from_graph,
    relation_embeddings_from_relation_types,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and cache relation embeddings from a graph using glossary-first semantic prompts."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV, TSV, or pickled networkx.Graph"
    )
    parser.add_argument(
        "--input-format", choices=("auto", "csv", "pickle"), default="auto"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output .pt cache path"
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=32,
        help="Output width for random relation embeddings. Semantic embeddings keep their full width.",
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
    parser.add_argument("--glossary-path", type=Path, default=DEFAULT_GLOSSARY_PATH)
    parser.add_argument(
        "--embedding-model",
        type=str,
        choices=["openai", "sapbert", "random"],
        default="openai",
        help="Embedding model: openai (text-embedding-3-small), sapbert (PubMedBERT-based), or random",
    )
    parser.add_argument(
        "--relation-text-mode",
        type=str,
        choices=["raw", "canonical"],
        default="raw",
        help="Text mode for relation embedding: raw (original glossary text) or canonical (rewritten clean sentences)",
    )
    parser.add_argument(
        "--sapbert-model",
        type=str,
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
        help="HuggingFace model name for SapBERT",
    )
    parser.add_argument(
        "--strict-embedding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of falling back if embedding fails.",
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


def _resolve_input_format(args: argparse.Namespace) -> str:
    input_format = args.input_format
    if input_format == "auto":
        suffix = args.input.suffix.lower()
        input_format = "pickle" if suffix in {".pkl", ".pickle"} else "csv"
    return input_format


def _load_relation_types_from_table(args: argparse.Namespace) -> list[str]:
    spec = _csv_spec_from_args(args)
    delimiter = _infer_delimiter(args.input, spec.delimiter)
    relation_col = spec.relation_col
    if _has_header(args.input, delimiter=delimiter, spec=spec):
        frame = pd.read_csv(
            args.input,
            sep=delimiter,
            usecols=[relation_col],
            nrows=args.max_edges,
            low_memory=False,
        )
        values = frame[relation_col].astype(str).str.strip()
    else:
        frame = pd.read_csv(
            args.input,
            sep=delimiter,
            header=None,
            usecols=[1],
            names=[spec.source_col, relation_col, spec.target_col],
            nrows=args.max_edges,
            low_memory=False,
        )
        values = frame[relation_col].astype(str).str.strip()
    return sorted(set(values.tolist()))


def main() -> None:
    args = parse_args()
    input_format = _resolve_input_format(args)
    if input_format == "pickle":
        graph = load_pickled_graph(args.input)
        edge_types = get_edge_types(graph)
        rel_emb = relation_embeddings_from_graph(
            graph,
            edge_dim=args.edge_dim,
            cache_path=args.output,
            embedding_model=args.embedding_model,
            relation_text_mode=args.relation_text_mode,
            strict_embedding=args.strict_embedding,
            glossary_path=args.glossary_path,
            sapbert_model=args.sapbert_model,
        )
    else:
        edge_types = _load_relation_types_from_table(args)
        rel_emb = relation_embeddings_from_relation_types(
            edge_types,
            edge_dim=args.edge_dim,
            cache_path=args.output,
            embedding_model=args.embedding_model,
            relation_text_mode=args.relation_text_mode,
            strict_embedding=args.strict_embedding,
            glossary_path=args.glossary_path,
            sapbert_model=args.sapbert_model,
        )
    print(f"Saved {len(rel_emb)} relation embeddings to {args.output}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Relation text mode: {args.relation_text_mode}")
    if args.embedding_model == "sapbert":
        print(f"SapBERT model: {args.sapbert_model}")
    print(f"Glossary path: {args.glossary_path}")
    # Print embedding dimension
    if len(rel_emb) > 0:
        sample_key = next(iter(rel_emb.keys()))
        print(f"Embedding dimension: {int(rel_emb[sample_key].numel())}")


if __name__ == "__main__":
    main()
