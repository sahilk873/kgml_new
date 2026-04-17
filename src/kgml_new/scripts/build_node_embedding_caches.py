from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


#
# Env bootstrap (must run before importing HF/SentenceTransformers/OpenAI helpers).
#

load_dotenv()

# Back-compat: user might set a misspelled variable in `.env`.
if os.environ.get("HUGGINGFACE_HUB_TOKEN") is None and os.environ.get("HF_TOKEN") is None:
    legacy = os.environ.get("HUGGINFACE_API_KEY")  # common typo (missing 'g')
    if legacy:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = legacy
        os.environ["HF_TOKEN"] = legacy

# Avoid accidental use of an expired/stale Hugging Face token from disk on shared clusters.
# If the user provided an explicit token via env/.env, allow libraries to use it.
has_explicit_hf_token = bool(
    os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)
if has_explicit_hf_token:
    os.environ.pop("HF_HUB_DISABLE_IMPLICIT_TOKEN", None)
else:
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from kgml_new.embeddings.semantic import _embed_async, _embed_e5, _embed_gemini


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_path: Path
    input_format: str
    source_col: str
    target_col: str
    source_type_col: str | None
    target_type_col: str | None
    has_header: bool


DEFAULT_DATASETS: dict[str, DatasetSpec] = {
    "drkg": DatasetSpec(
        name="drkg",
        input_path=Path("drkg.tsv"),
        input_format="tsv",
        source_col="source",
        target_col="target",
        source_type_col=None,
        target_type_col=None,
        has_header=False,
    ),
    "primekg": DatasetSpec(
        name="primekg",
        input_path=Path("data/kg.csv"),
        input_format="csv",
        source_col="x_name",
        target_col="y_name",
        source_type_col="x_type",
        target_type_col="y_type",
        has_header=True,
    ),
    "fb15k237": DatasetSpec(
        name="fb15k237",
        input_path=Path("data/fb15k-237/fb15k-237.tsv"),
        input_format="tsv",
        source_col="source",
        target_col="target",
        source_type_col=None,
        target_type_col=None,
        has_header=False,
    ),
}

DEFAULT_EMBEDDING_MODELS = ("openai", "e5", "gemini")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable node embedding caches for DRKG/PrimeKG/FB15k-237."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_DATASETS.keys()),
        default=list(DEFAULT_DATASETS.keys()),
        help="Datasets to process (default: all).",
    )
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        choices=list(DEFAULT_EMBEDDING_MODELS),
        default=list(DEFAULT_EMBEDDING_MODELS),
        help="Embedding backends (default: openai e5 gemini).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for node embedding cache files.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--e5-model",
        type=str,
        default="intfloat/e5-base-v2",
        help="E5 model name.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-embedding-001",
        help="Gemini embedding model name.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing cache files.",
    )
    parser.add_argument(
        "--openai-batch-size",
        type=int,
        default=256,
        help="Number of node texts per OpenAI embeddings request.",
    )
    parser.add_argument(
        "--e5-batch-size",
        type=int,
        default=1024,
        help="Number of node texts per E5 encode call.",
    )
    parser.add_argument(
        "--gemini-batch-size",
        type=int,
        default=100,
        help="Number of node texts per Gemini embed call (API limit is 100).",
    )
    return parser.parse_args(argv)


def _read_table(spec: DatasetSpec) -> pd.DataFrame:
    sep = "\t" if spec.input_format == "tsv" else ","
    if spec.has_header:
        return pd.read_csv(spec.input_path, sep=sep, low_memory=False)
    return pd.read_csv(
        spec.input_path,
        sep=sep,
        header=None,
        names=[spec.source_col, "relation", spec.target_col],
        usecols=[0, 1, 2],
        low_memory=False,
    )


def _infer_type_from_node_id(node_id: str) -> str:
    if "::" in node_id:
        prefix = node_id.split("::", maxsplit=1)[0].strip()
        if prefix:
            return prefix
    return "entity"


def _human_readable_node_id(node_id: str) -> str:
    if "::" in node_id:
        suffix = node_id.split("::")[-1].strip()
        if suffix:
            return suffix
    return node_id.strip()


def _extract_nodes_and_types(spec: DatasetSpec) -> tuple[list[str], list[str], list[str]]:
    frame = _read_table(spec)
    node_to_type: dict[str, str] = {}

    for row in frame.itertuples(index=False):
        src = str(getattr(row, spec.source_col)).strip()
        dst = str(getattr(row, spec.target_col)).strip()
        if not src or not dst:
            continue

        if spec.source_type_col is not None:
            src_type = str(getattr(row, spec.source_type_col)).strip()
        else:
            src_type = _infer_type_from_node_id(src)
        if spec.target_type_col is not None:
            dst_type = str(getattr(row, spec.target_type_col)).strip()
        else:
            dst_type = _infer_type_from_node_id(dst)

        node_to_type.setdefault(src, src_type or "entity")
        node_to_type.setdefault(dst, dst_type or "entity")

    node_ids = sorted(node_to_type.keys())
    node_types = [node_to_type[node_id] for node_id in node_ids]
    node_texts = [
        f"Node: {_human_readable_node_id(node_id)}. Type: {node_type}."
        for node_id, node_type in zip(node_ids, node_types, strict=True)
    ]
    return node_ids, node_types, node_texts


def _embed_openai(texts: list[str], model_name: str) -> list[list[float]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    return asyncio.run(_embed_async(client, texts, model=model_name))


def _chunked(items: list[str], chunk_size: int):
    for idx in range(0, len(items), chunk_size):
        yield idx, items[idx : idx + chunk_size]


def _embed_texts(
    *,
    texts: list[str],
    embedding_model: str,
    openai_model: str,
    e5_model: str,
    gemini_model: str,
    openai_batch_size: int,
    e5_batch_size: int,
    gemini_batch_size: int,
) -> torch.Tensor:
    vectors: list[list[float]] = []
    if embedding_model == "openai":
        for start_idx, chunk in _chunked(texts, openai_batch_size):
            print(
                f"[node-cache] openai batch start={start_idx} size={len(chunk)}",
                flush=True,
            )
            vectors.extend(_embed_openai(chunk, openai_model))
    elif embedding_model == "e5":
        for start_idx, chunk in _chunked(texts, e5_batch_size):
            print(
                f"[node-cache] e5 batch start={start_idx} size={len(chunk)}",
                flush=True,
            )
            vectors.extend(_embed_e5(chunk, model_name=e5_model))
    elif embedding_model == "gemini":
        for start_idx, chunk in _chunked(texts, gemini_batch_size):
            print(
                f"[node-cache] gemini batch start={start_idx} size={len(chunk)}",
                flush=True,
            )
            vectors.extend(_embed_gemini(chunk, model_name=gemini_model))
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")
    return torch.tensor(vectors, dtype=torch.float32)


def _cache_path(output_dir: Path, dataset_name: str, embedding_model: str) -> Path:
    return output_dir / f"{dataset_name}-node-embeddings-{embedding_model}.pt"


def run(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for dataset_name in args.datasets:
        spec = DEFAULT_DATASETS[dataset_name]
        if not spec.input_path.is_file():
            raise FileNotFoundError(
                f"Missing input for dataset '{dataset_name}': {spec.input_path}"
            )
        print(
            f"[node-cache] start dataset={dataset_name} input={spec.input_path}",
            flush=True,
        )
        node_ids, node_types, node_texts = _extract_nodes_and_types(spec)
        if not node_ids:
            raise ValueError(f"No nodes parsed from {spec.input_path}")

        print(
            f"[node-cache] dataset={dataset_name} input={spec.input_path} "
            f"nodes={len(node_ids)}"
        )

        for embedding_model in args.embedding_models:
            out_path = _cache_path(output_dir, dataset_name, embedding_model)
            if out_path.exists() and not args.overwrite:
                print(f"[node-cache] skip existing {out_path}")
                written.append(out_path)
                continue

            embeddings = _embed_texts(
                texts=node_texts,
                embedding_model=embedding_model,
                openai_model=args.openai_model,
                e5_model=args.e5_model,
                gemini_model=args.gemini_model,
                openai_batch_size=args.openai_batch_size,
                e5_batch_size=args.e5_batch_size,
                gemini_batch_size=args.gemini_batch_size,
            )
            payload = {
                "format_version": 1,
                "artifact_type": "node_embeddings",
                "dataset": dataset_name,
                "embedding_model": embedding_model,
                "embedding_dim": int(embeddings.size(1)),
                "num_nodes": int(embeddings.size(0)),
                "node_text_mode": "human_readable_id_plus_type",
                "node_ids": node_ids,
                "node_types": node_types,
                "node_texts": node_texts,
                # run_gpu_method can read this directly via --node-embeddings-path
                "embeddings": embeddings,
            }
            torch.save(payload, out_path)
            written.append(out_path)
            print(
                f"[node-cache] wrote dataset={dataset_name} model={embedding_model} "
                f"shape=({payload['num_nodes']}, {payload['embedding_dim']}) "
                f"path={out_path}"
            )
    return written


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
