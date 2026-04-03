from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from kgml_new.data.relations import RELATION_DESCRIPTIONS, get_edge_types


DEFAULT_GLOSSARY_PATH = Path(__file__).resolve().parents[3] / "relation_glossary.tsv"


@lru_cache(maxsize=4)
def _load_relation_glossary(glossary_path: str) -> dict[str, dict[str, str]]:
    path = Path(glossary_path)
    if not path.exists():
        return {}

    glossary: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            relation_name = str(row.get("Relation-name", "")).strip()
            if relation_name:
                glossary[relation_name] = {key: str(value).strip() for key, value in row.items() if value is not None}
    return glossary


def _should_use_glossary(rel: str) -> bool:
    # DRKG-style sourced predicates carry namespace separators like
    # `DRUGBANK::target::Compound:Gene` or `bioarx::HumGenHumGen:Gene:Gene`.
    # PrimeKG-style labels such as `drug_protein` should stay as raw strings.
    return "::" in rel


def describe_relation(rel: str, *, glossary_path: Path | None = None) -> str:
    if _should_use_glossary(rel):
        glossary = _load_relation_glossary(str(glossary_path or DEFAULT_GLOSSARY_PATH))
        entry = glossary.get(rel)
        if entry:
            parts = [f"Relation name: {rel}."]
            data_source = entry.get("Data-source")
            entity_types = entry.get("Connected entity-types")
            interaction_type = entry.get("Interaction-type")
            description = entry.get("Description")
            if data_source:
                parts.append(f"Data source: {data_source}.")
            if entity_types:
                parts.append(f"Connected entity types: {entity_types}.")
            if interaction_type:
                parts.append(f"Interaction type: {interaction_type}.")
            if description and description.lower() != "nan":
                parts.append(f"Description: {description}.")
            return " ".join(parts)

    if rel in RELATION_DESCRIPTIONS:
        return RELATION_DESCRIPTIONS[rel]
    return f"Relationship type: {rel}"


async def _embed_async(
    client, texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    response = await client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in response.data]


def _infer_embedding_dim(rel_emb: dict[str, torch.Tensor], default: int) -> int:
    for value in rel_emb.values():
        return int(value.numel())
    return default


def _load_cached_relation_embeddings(
    cache_path: Path,
    *,
    use_openai: bool,
    edge_dim: int,
) -> dict[str, torch.Tensor] | None:
    with cache_path.open("rb") as f:
        cached = torch.load(f)

    if isinstance(cached, dict) and "embeddings" in cached:
        embeddings = cached["embeddings"]
        if isinstance(embeddings, dict):
            return embeddings

    if not isinstance(cached, dict):
        return None

    # Legacy caches were stored as a plain relation->tensor mapping. If a
    # semantic cache still has message-space width, it predates the full-width
    # semantic fix and must be regenerated.
    inferred_dim = _infer_embedding_dim(cached, edge_dim)
    if use_openai and inferred_dim <= edge_dim:
        print(
            f"Regenerating legacy semantic cache at {cache_path} "
            f"(cached dim={inferred_dim}, edge_dim={edge_dim})"
        )
        return None
    return cached


def relation_embeddings_from_relation_types(
    relation_types: list[str],
    *,
    edge_dim: int = 32,
    cache_path: Path | None = None,
    use_openai: bool = True,
    strict_openai: bool = False,
    glossary_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    if cache_path and cache_path.exists():
        cached = _load_cached_relation_embeddings(
            cache_path,
            use_openai=use_openai,
            edge_dim=edge_dim,
        )
        if cached is not None:
            return cached

    rel_types = sorted(set(relation_types))
    rel_emb: dict[str, torch.Tensor] = {}

    if use_openai:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            texts = [describe_relation(rel, glossary_path=glossary_path) for rel in rel_types]
            import asyncio

            embeddings = asyncio.run(_embed_async(client, texts))
            for rel, emb in zip(rel_types, embeddings):
                # Keep the full semantic embedding frozen and let the model
                # learn a projection into the edge-message space.
                rel_emb[rel] = torch.tensor(emb, dtype=torch.float32)
        except ModuleNotFoundError as e:
            if strict_openai:
                raise RuntimeError(
                    "OpenAI dependency missing while strict_openai=True"
                ) from e
            print(f"OpenAI dependency missing ({e}); using random embeddings")
            use_openai = False
        except Exception as e:
            if strict_openai:
                raise RuntimeError(
                    "OpenAI embedding failed while strict_openai=True"
                ) from e
            print(f"OpenAI embedding failed: {e}, using random embeddings")
            use_openai = False

    if not use_openai:
        rng = torch.Generator()
        rng.manual_seed(42)
        for rel in rel_types:
            rel_emb[rel] = (
                torch.randn(edge_dim, generator=rng, dtype=torch.float32) * 0.1
            )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            torch.save(
                {
                    "format_version": 2,
                    "use_openai": use_openai,
                    "embedding_dim": _infer_embedding_dim(rel_emb, edge_dim),
                    "embeddings": rel_emb,
                },
                f,
            )
        print(f"Cached relation embeddings to {cache_path}")

    return rel_emb


def relation_embeddings_from_graph(
    graph,
    edge_dim: int = 32,
    cache_path: Path | None = None,
    use_openai: bool = True,
    strict_openai: bool = False,
    glossary_path: Path | None = None,
) -> dict[str, torch.Tensor]:
    """
    Get relation embeddings from graph edges using OpenAI embeddings.

    If cache_path is provided, loads from cache or saves to cache.
    If use_openai is False, returns random embeddings (for ablation).
    If strict_openai is True and OpenAI embedding setup fails, raises RuntimeError.
    """
    rel_types = get_edge_types(graph)
    return relation_embeddings_from_relation_types(
        rel_types,
        edge_dim=edge_dim,
        cache_path=cache_path,
        use_openai=use_openai,
        strict_openai=strict_openai,
        glossary_path=glossary_path,
    )


def build_relation_tensor(
    rel_emb: dict[str, torch.Tensor],
    relation_lookup: dict[str, int],
    edge_dim: int,
    device: torch.device,
) -> torch.Tensor:
    max_idx = max(relation_lookup.values()) + 1
    inferred_dim = _infer_embedding_dim(rel_emb, edge_dim)
    table = torch.zeros(max_idx, inferred_dim, dtype=torch.float32, device=device)
    for rel, idx in relation_lookup.items():
        if rel in rel_emb:
            table[idx] = rel_emb[rel]
    return table
