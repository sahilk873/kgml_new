from __future__ import annotations

import csv
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


load_dotenv()

from kgml_new.data.relations import RELATION_DESCRIPTIONS, get_edge_types


DEFAULT_GLOSSARY_PATH = Path(__file__).resolve().parents[3] / "relation_glossary.tsv"

EmbeddingModel = Literal["openai", "gemini", "sapbert", "e5", "random"]
RelationTextMode = Literal["raw", "canonical"]


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
                glossary[relation_name] = {
                    key: str(value).strip()
                    for key, value in row.items()
                    if value is not None
                }
    return glossary


def _should_use_glossary(rel: str) -> bool:
    # DRKG-style sourced predicates carry namespace separators like
    # `DRUGBANK::target::Compound:Gene` or `bioarx::HumGenHumGen:Gene:Gene`.
    # PrimeKG-style labels such as `drug_protein` should stay as raw strings.
    return "::" in rel


def describe_relation(
    rel: str,
    *,
    glossary_path: Path | None = None,
    relation_text_mode: RelationTextMode = "raw",
    use_canonical_cache: bool = True,
) -> str:
    """Get text for embedding a relation.

    Args:
        rel: Relation name
        glossary_path: Path to relation glossary
        relation_text_mode: "raw" for original glossary text, "canonical" for rewritten clean text
        use_canonical_cache: Whether to use cached canonical text (if available)
    """
    if relation_text_mode == "canonical":
        # Determine cache path for canonical text
        base_path = glossary_path or DEFAULT_GLOSSARY_PATH
        canonical_cache = base_path.parent / f"{base_path.stem}.canonical.txt"
        return get_canonical_relation_text(
            rel,
            glossary_path=glossary_path,
            canonical_cache_path=canonical_cache if use_canonical_cache else None,
        )

    # Default: raw glossary text
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


def _extract_entity_types(entity_str: str) -> tuple[str, str]:
    """Extract source and target entity types from 'Compound:Gene' format."""
    if not entity_str or entity_str.lower() == "nan":
        return "entity", "entity"
    parts = entity_str.split(":")
    if len(parts) >= 2:
        return parts[0].lower(), parts[1].lower()
    return "entity", "entity"


def _canonical_relation_text(
    rel: str,
    interaction_type: str,
    entity_types: str,
    description: str,
) -> str:
    """Rewrite a relation into a clean canonical biomedical sentence.

    Properties:
    - No source tags (DGIDB::, STRING::, etc.)
    - Clean entity types
    - Explicit direction/effect words
    - Single concise sentence
    """
    src_type, tgt_type = _extract_entity_types(entity_types)

    # Map interaction types to canonical direction words
    direction_map = {
        "activation": "increases",
        "activates": "increases",
        "activator": "increases",
        "inhibition": "decreases",
        "inhibits": "decreases",
        "inhibitor": "decreases",
        "agonism": "activates",
        "agonist": "activates",
        "antagonism": "blocks",
        "antagonist": "blocks",
        "blocking": "blocks",
        "blocker": "blocks",
        "partial agonism": "partially activates",
        "allosteric modulation": "modulates",
        "positive allosteric modulation": "increases",
        "negative allosteric modulation": "decreases",
        "upregulation": "upregulates",
        "downregulation": "downregulates",
        "increases expression/production": "increases expression of",
        "decreases expression/production": "decreases expression of",
        "affects expression/production (neutral)": "modulates",
        "binding": "binds to",
        "binding, ligand (esp. receptors)": "binds to",
        "catalysis": "catalyzes",
        "cleavage reaction": "cleaves",
        "colocalization": "colocalizes with",
        "dephosphorylation reaction": "dephosphorylates",
        "direct interaction": "directly interacts with",
        "association": "associates with",
        "physical association": "physically associates with",
        "proteolysis": "cleaves",
        "ubiquitination": "ubiquitinates",
        "expression": "affects expression of",
        "regulation": "regulates",
        "participation": "participates in",
        "same protein or complex": "is part of the same complex as",
        "signaling pathway": "participates in signaling pathway",
        "carrier": "is a carrier for",
        "enzyme": "is an enzyme for",
        "target": "is a target of",
        "drug-drug interaction": "interacts with",
        "Compound treats the disease": "treats",
        "causes": "causes",
        "palliation": "palliates",
        "treatment": "treats",
        "localization": "localizes to",
        "presents": "presents",
        "covariation": "covaries with",
        "resemblence": "resembles",
        "resemblance": "resembles",
        "in_tax": "belongs to taxonomy",
    }

    # Get direction word
    direction = direction_map.get(interaction_type.lower(), interaction_type.lower())

    # Handle entity type mappings for cleaner sentences
    entity_map = {
        "compound": "compound",
        "gene": "gene product",
        "disease": "disease",
        "anatomy": "anatomical location",
        "symptom": "symptom",
        "side effect": "side effect",
        "pathway": "pathway",
        "cellular component": "cellular component",
        "biological process": "biological process",
        "molecular function": "molecular function",
    }

    src = entity_map.get(src_type, src_type)
    tgt = entity_map.get(tgt_type, tgt_type)

    # Build canonical sentence based on interaction type
    templates = {
        "increases": f"A {src} increases the activity or function of a {tgt}.",
        "decreases": f"A {src} decreases the activity or function of a {tgt}.",
        "activates": f"A {src} activates a {tgt}.",
        "blocks": f"A {src} blocks or inhibits a {tgt}.",
        "partially activates": f"A {src} partially activates a {tgt}.",
        "modulates": f"A {src} modulates the function of a {tgt}.",
        "upregulates": f"A {tgt} is upregulated in a {src}.",
        "downregulates": f"A {tgt} is downregulated in a {src}.",
        "increases expression of": f"A {src} increases expression of a {tgt}.",
        "decreases expression of": f"A {src} decreases expression of a {tgt}.",
        "binds to": f"A {src} binds to a {tgt}.",
        "catalyzes": f"A {src} catalyzes a reaction involving a {tgt}.",
        "cleaves": f"An enzyme cleaves a {tgt}.",
        "dephosphorylates": f"An enzyme dephosphorylates a {tgt}.",
        "directly interacts with": f"A {src} directly interacts with a {tgt}.",
        "associates with": f"A {src} associates with a {tgt}.",
        "physically associates with": f"A {src} physically associates with a {tgt}.",
        "ubiquitinates": f"A {src} ubiquitinates a {tgt}.",
        "affects expression of": f"A {src} affects expression of a {tgt}.",
        "regulates": f"A {src} regulates a {tgt}.",
        "participates in": f"A {src} participates in a {tgt}.",
        "is part of the same complex as": f"A {src} is part of the same complex as a {tgt}.",
        "participates in signaling pathway": f"A {src} participates in a signaling pathway.",
        "is a carrier for": f"A {src} is a carrier for a {tgt}.",
        "is an enzyme for": f"A {src} is an enzyme for a {tgt}.",
        "is a target of": f"A {src} is a target of a {tgt}.",
        "interacts with": f"A {src} interacts with a {tgt}.",
        "treats": f"A {src} treats a {tgt}.",
        "causes": f"A {src} causes a {tgt}.",
        "palliates": f"A {src} palliates a {tgt}.",
        "localizes to": f"A {src} localizes to a {tgt}.",
        "covaries with": f"A {src} covaries with a {tgt}.",
        "resembles": f"A {src} resembles a {tgt}.",
        "belongs to taxonomy": f"A {src} belongs to a taxon.",
    }

    template = templates.get(direction)
    if template:
        return template

    # Fallback: try to use description if available
    if description and description.lower() != "nan":
        # Clean description: lowercase, remove special chars
        clean_desc = re.sub(r"[^\w\s]", " ", description.lower())
        clean_desc = re.sub(r"\s+", " ", clean_desc).strip()
        # Limit length
        if len(clean_desc) > 100:
            clean_desc = clean_desc[:100] + "..."
        return f"A {src} {clean_desc}."

    # Ultimate fallback
    return f"A {src} {direction} a {tgt}."


def _load_canonical_text_cache(cache_path: Path) -> dict[str, str] | None:
    """Load cached canonical relation text."""
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return {
                line.split("\t")[0]: line.split("\t")[1] for line in f if "\t" in line
            }
    except Exception:
        return None


def _save_canonical_text_cache(
    cache_path: Path, canonical_texts: dict[str, str]
) -> None:
    """Save canonical relation text to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        for rel, text in canonical_texts.items():
            f.write(f"{rel}\t{text}\n")


def get_canonical_relation_text(
    rel: str,
    glossary_path: Path | None = None,
    canonical_cache_path: Path | None = None,
) -> str:
    """Get canonical rewritten text for a relation.

    Uses cached canonical texts if available, otherwise generates from glossary.
    The canonical text is a clean, mechanism-focused sentence.
    """
    # Try cache first
    cache = None
    if canonical_cache_path:
        cache = _load_canonical_text_cache(canonical_cache_path)
        if cache and rel in cache:
            return cache[rel]

    # Get glossary entry
    glossary = _load_relation_glossary(str(glossary_path or DEFAULT_GLOSSARY_PATH))
    entry = glossary.get(rel, {})

    interaction_type = entry.get("Interaction-type", "")
    entity_types = entry.get("Connected entity-types", "")
    description = entry.get("Description", "")

    canonical = _canonical_relation_text(
        rel, interaction_type, entity_types, description
    )

    # Cache if path provided
    if canonical_cache_path:
        all_canonical = cache or {}
        all_canonical[rel] = canonical
        _save_canonical_text_cache(canonical_cache_path, all_canonical)

    return canonical


async def _embed_async(
    client, texts: list[str], model: str = "text-embedding-3-small"
) -> list[list[float]]:
    response = await client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in response.data]


def _gemini_embed_response_to_vectors(response) -> list[list[float]]:
    if hasattr(response, "embeddings") and response.embeddings is not None:
        return [list(item.values) for item in response.embeddings]
    if hasattr(response, "embedding") and response.embedding is not None:
        return [list(response.embedding.values)]
    raise RuntimeError("Gemini embedding response did not include embeddings.")


def _gemini_embed_chunk_with_retries(client, model: str, chunk: list[str]):
    """Call embed_content with backoff on rate limits."""
    last_exc: BaseException | None = None
    for attempt in range(20):
        try:
            return client.models.embed_content(model=model, contents=chunk)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                delay = min(3.0 * (1.35**attempt), 120.0)
                time.sleep(delay)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _embed_gemini(
    texts: list[str], model_name: str = "gemini-3.1-flash-lite-preview"
) -> list[list[float]]:
    """Generate embeddings via Gemini API."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set; add it to your environment or .env file."
        )

    client = genai.Client(api_key=api_key)
    requested_model = model_name
    effective_model = model_name
    # BatchEmbedContents allows at most 100 requests per call.
    batch_size = 100
    merged: list[list[float]] = []

    for batch_idx, start in enumerate(range(0, len(texts), batch_size)):
        if batch_idx > 0:
            # Free tier embed quotas are tight; avoid back-to-back bursts across batches.
            time.sleep(65.0)
        chunk = texts[start : start + batch_size]
        if batch_idx == 0:
            try:
                response = _gemini_embed_chunk_with_retries(
                    client, effective_model, chunk
                )
            except Exception as exc:
                message = str(exc)
                if (
                    "not supported for embedContent" in message
                    or "is not found for API version" in message
                ) and requested_model != "gemini-embedding-001":
                    # Gemini Flash models are generation models; embedContent expects
                    # an embedding-capable model such as gemini-embedding-001.
                    effective_model = "gemini-embedding-001"
                    print(
                        f"Gemini model '{requested_model}' is not embedding-capable; "
                        f"retrying with '{effective_model}'."
                    )
                    response = _gemini_embed_chunk_with_retries(
                        client, effective_model, chunk
                    )
                else:
                    raise
        else:
            response = _gemini_embed_chunk_with_retries(
                client, effective_model, chunk
            )
        merged.extend(_gemini_embed_response_to_vectors(response))

    return merged


def _embed_sapbert(
    texts: list[str],
    model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
) -> list[list[float]]:
    """Generate embeddings using SapBERT (PubMedBERT-based biomedical embeddings)."""
    from sentence_transformers import SentenceTransformer

    # Prefer an explicit token from the environment (for clusters that require auth
    # even for public downloads), but avoid implicit cached tokens that might be stale.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or False
    model = SentenceTransformer(model_name, token=token)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def _embed_e5(
    texts: list[str], model_name: str = "intfloat/e5-base-v2"
) -> list[list[float]]:
    """Generate embeddings using E5 (SentenceTransformers-compatible HuggingFace model)."""
    from sentence_transformers import SentenceTransformer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or False
    model = SentenceTransformer(model_name, token=token)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


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
    embedding_model: EmbeddingModel = "openai",
    relation_text_mode: RelationTextMode = "raw",
    strict_embedding: bool = False,
    glossary_path: Path | None = None,
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
    e5_model: str = "intfloat/e5-base-v2",
    gemini_model: str = "gemini-3.1-flash-lite-preview",
) -> dict[str, torch.Tensor]:
    """Generate relation embeddings using specified model.

    Args:
        relation_types: List of relation type strings
        edge_dim: Dimension for random embeddings
        cache_path: Path to save/load cache
        embedding_model: Model to use - "openai", "gemini", "sapbert", "e5", or "random"
        relation_text_mode: "raw" for original glossary text, "canonical" for rewritten clean text
        strict_embedding: If True, raise error on embedding failure
        glossary_path: Path to relation glossary
        sapbert_model: HuggingFace model name for SapBERT
        e5_model: HuggingFace model name for E5
    """
    use_openai = embedding_model == "openai"
    use_gemini = embedding_model == "gemini"
    use_sapbert = embedding_model == "sapbert"
    use_e5 = embedding_model == "e5"
    use_random = embedding_model == "random"

    # Check cache
    if cache_path and cache_path.exists():
        with cache_path.open("rb") as f:
            cached = torch.load(f)

        if isinstance(cached, dict) and "embeddings" in cached:
            cached_emb = cached["embeddings"]
            if isinstance(cached_emb, dict):
                # Verify cache matches requested model type
                cached_model = cached.get("embedding_model", "unknown")
                cached_text_mode = cached.get("relation_text_mode", "raw")
                if (
                    cached_model == embedding_model
                    and cached_text_mode == relation_text_mode
                ):
                    return cached_emb
                else:
                    print(
                        f"Cache exists but was generated with model={cached_model}, text_mode={cached_text_mode}, "
                        f"regenerating with model={embedding_model}, text_mode={relation_text_mode}"
                    )

    rel_types = sorted(set(relation_types))
    rel_emb: dict[str, torch.Tensor] = {}

    # Get texts for embedding based on text mode
    texts = [
        describe_relation(
            rel,
            glossary_path=glossary_path,
            relation_text_mode=relation_text_mode,
        )
        for rel in rel_types
    ]

    # Log sample texts for debugging
    if relation_text_mode == "canonical":
        print(f"=== Canonical Text Mode Samples ===")
        for i, (rel, text) in enumerate(zip(rel_types[:5], texts[:5])):
            print(f"{rel[:40]:40s} -> {text}")

    if use_sapbert:
        try:
            # texts already prepared above
            embeddings = _embed_sapbert(texts, model_name=sapbert_model)
            for rel, emb in zip(rel_types, embeddings):
                rel_emb[rel] = torch.tensor(emb, dtype=torch.float32)
            print(
                f"Generated SapBERT embeddings ({sapbert_model}) for {len(rel_types)} relations"
            )
        except Exception as e:
            if strict_embedding:
                raise RuntimeError(
                    f"SapBERT embedding failed while strict_embedding=True: {e}"
                ) from e
            print(f"SapBERT embedding failed: {e}, using random embeddings")
            use_random = True

    if use_e5:
        try:
            embeddings = _embed_e5(texts, model_name=e5_model)
            for rel, emb in zip(rel_types, embeddings):
                rel_emb[rel] = torch.tensor(emb, dtype=torch.float32)
            print(f"Generated E5 embeddings ({e5_model}) for {len(rel_types)} relations")
        except Exception as e:
            if strict_embedding:
                raise RuntimeError(
                    f"E5 embedding failed while strict_embedding=True: {e}"
                ) from e
            print(f"E5 embedding failed: {e}, using random embeddings")
            use_random = True

    if use_openai:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            # texts already prepared above
            import asyncio

            embeddings = asyncio.run(_embed_async(client, texts))
            for rel, emb in zip(rel_types, embeddings):
                rel_emb[rel] = torch.tensor(emb, dtype=torch.float32)
            print(f"Generated OpenAI embeddings for {len(rel_types)} relations")
        except ModuleNotFoundError as e:
            if strict_embedding:
                raise RuntimeError(
                    "OpenAI dependency missing while strict_embedding=True"
                ) from e
            print(f"OpenAI dependency missing ({e}); using random embeddings")
            use_random = True
        except Exception as e:
            if strict_embedding:
                raise RuntimeError(
                    f"OpenAI embedding failed while strict_embedding=True: {e}"
                ) from e
            print(f"OpenAI embedding failed: {e}, using random embeddings")
            use_random = True

    if use_gemini:
        try:
            embeddings = _embed_gemini(texts, model_name=gemini_model)
            for rel, emb in zip(rel_types, embeddings):
                rel_emb[rel] = torch.tensor(emb, dtype=torch.float32)
            print(
                f"Generated Gemini embeddings ({gemini_model}) for {len(rel_types)} relations"
            )
        except ModuleNotFoundError as e:
            if strict_embedding:
                raise RuntimeError(
                    "Gemini dependency missing while strict_embedding=True"
                ) from e
            print(f"Gemini dependency missing ({e}); using random embeddings")
            use_random = True
        except Exception as e:
            if strict_embedding:
                raise RuntimeError(
                    f"Gemini embedding failed while strict_embedding=True: {e}"
                ) from e
            print(f"Gemini embedding failed: {e}, using random embeddings")
            use_random = True

    if use_random:
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
                    "format_version": 4,
                    "embedding_model": embedding_model,
                    "relation_text_mode": relation_text_mode,
                    "sapbert_model": sapbert_model if use_sapbert else None,
                    "e5_model": e5_model if use_e5 else None,
                    "gemini_model": gemini_model if use_gemini else None,
                    "use_openai": use_openai,
                    "use_gemini": use_gemini,
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
    embedding_model: EmbeddingModel = "openai",
    relation_text_mode: RelationTextMode = "raw",
    strict_embedding: bool = False,
    glossary_path: Path | None = None,
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token",
    e5_model: str = "intfloat/e5-base-v2",
    gemini_model: str = "gemini-3.1-flash-lite-preview",
) -> dict[str, torch.Tensor]:
    """
    Get relation embeddings from graph edges using specified embedding model.

    If cache_path is provided, loads from cache or saves to cache.
    embedding_model: "openai", "gemini", "sapbert", "e5", or "random"
    relation_text_mode: "raw" for original glossary text, "canonical" for rewritten clean text
    If strict_embedding is True and embedding fails, raises RuntimeError.
    """
    rel_types = get_edge_types(graph)
    return relation_embeddings_from_relation_types(
        rel_types,
        edge_dim=edge_dim,
        cache_path=cache_path,
        embedding_model=embedding_model,
        relation_text_mode=relation_text_mode,
        strict_embedding=strict_embedding,
        glossary_path=glossary_path,
        sapbert_model=sapbert_model,
        e5_model=e5_model,
        gemini_model=gemini_model,
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


def compute_semantic_similarity_matrix(
    relation_embeddings: dict[str, torch.Tensor],
    relation_lookup: dict[str, int],
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix from relation embeddings.

    Returns:
        similarity_matrix: (num_relations, num_relations) tensor where
            similarity_matrix[i,j] = cos(semantic_embedding_i, semantic_embedding_j)
    """
    num_relations = max(relation_lookup.values()) + 1

    embeddings_list = []
    rel_indices = []
    for rel, idx in relation_lookup.items():
        if rel in relation_embeddings:
            embeddings_list.append(relation_embeddings[rel])
            rel_indices.append(idx)

    if not embeddings_list:
        return torch.zeros(num_relations, num_relations)

    embeddings = torch.stack(embeddings_list)
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    similarity_matrix = torch.mm(embeddings, embeddings.t())

    full_matrix = torch.zeros(num_relations, num_relations)

    for i, rel_idx_i in enumerate(rel_indices):
        for j, rel_idx_j in enumerate(rel_indices):
            full_matrix[rel_idx_i, rel_idx_j] = similarity_matrix[i, j]

    return full_matrix
