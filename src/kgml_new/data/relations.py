"""Relation strings on NetworkX edges and optional PrimeKG-style glosses."""

from __future__ import annotations

import networkx as nx

_REL_KEYS = (
    "relationship",
    "predicate",
    "relationship_type",
    "label",
    "edge_type",
)

# Optional short descriptions for relation embedding text (extend per deployment).
RELATION_DESCRIPTIONS: dict[str, str] = {}


def get_edge_relation(edge_attrs: dict) -> str:
    """Return relation/predicate string from edge attribute dict."""
    for key in _REL_KEYS:
        if key in edge_attrs and edge_attrs[key] is not None:
            return str(edge_attrs[key])
    return "UNK"


def get_edge_types(graph: nx.Graph) -> list[str]:
    """Sorted unique relation strings on edges (via ``get_edge_relation``)."""
    types_set: set[str] = set()
    for _, _, attrs in graph.edges(data=True):
        types_set.add(get_edge_relation(attrs))
    return sorted(types_set)


def build_relation_lookup(relation_keys: list[str]) -> tuple[dict[str, int], int]:
    """Build a relation-id lookup with ``UNK`` fixed at index 0."""
    lookup: dict[str, int] = {"UNK": 0, "unk": 0}
    next_idx = 1
    for key in sorted(set(relation_keys)):
        normalized = str(key).strip()
        if not normalized:
            continue
        if normalized.upper() == "UNK":
            lookup[normalized] = 0
            lookup[normalized.lower()] = 0
            lookup[normalized.upper()] = 0
            continue
        idx = lookup.get(normalized)
        if idx is None:
            idx = next_idx
            next_idx += 1
        lookup.setdefault(normalized, idx)
        lookup.setdefault(normalized.lower(), idx)
        lookup.setdefault(normalized.upper(), idx)
    return lookup, 0
