from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

from kgml_new.data.relations import RELATION_DESCRIPTIONS, get_edge_types


def _default_description(rel: str) -> str:
    return RELATION_DESCRIPTIONS.get(rel, f"Relationship type: {rel}")


async def _embed_async(
    client, texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    response = await client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in response.data]


def relation_embeddings_from_graph(
    graph,
    edge_dim: int = 32,
    cache_path: Path | None = None,
    use_openai: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Get relation embeddings from graph edges using OpenAI embeddings.

    If cache_path is provided, loads from cache or saves to cache.
    If use_openai is False, returns random embeddings (for ablation).
    """
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = torch.load(f)
        return cached

    rel_types = get_edge_types(graph)
    rel_emb: dict[str, torch.Tensor] = {}

    if use_openai:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            texts = [_default_description(rel) for rel in rel_types]
            import asyncio

            embeddings = asyncio.run(_embed_async(client, texts))
            for rel, emb in zip(rel_types, embeddings):
                rel_emb[rel] = torch.tensor(emb[:edge_dim], dtype=torch.float32)
        except ModuleNotFoundError as e:
            print(f"OpenAI dependency missing ({e}); using random embeddings")
            use_openai = False
        except Exception as e:
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
            torch.save(rel_emb, f)
        print(f"Cached relation embeddings to {cache_path}")

    return rel_emb


def build_relation_tensor(
    rel_emb: dict[str, torch.Tensor],
    relation_lookup: dict[str, int],
    edge_dim: int,
    device: torch.device,
) -> torch.Tensor:
    max_idx = max(relation_lookup.values()) + 1
    table = torch.zeros(max_idx, edge_dim, dtype=torch.float32, device=device)
    for rel, idx in relation_lookup.items():
        if rel in rel_emb:
            table[idx] = rel_emb[rel]
    return table
