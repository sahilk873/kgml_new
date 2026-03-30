from __future__ import annotations

import networkx as nx

# PharmKG-style glosses; extend for PrimeKG / custom predicates as needed.
RELATION_DESCRIPTIONS: dict[str, str] = {
    "A": "Agonism, activation, or antagonism, blocking",
    "B": "Binding, ligand",
    "C": "Inhibits cell growth",
    "CC": "Relationship between chemicals",
    "D": "Possible therapeutic effect",
    "E": "Affects expression or production",
    "GG": "Relationship between genes",
    "Gene-Chemical": "Association between gene and chemical",
    "Gene-Disease": "Association between gene and disease",
    "Gene-Gene": "Relationship between genes",
    "H": "Unknown",
    "J": "Role in pathogenesis",
    "K": "Metabolism, pharmacokinetics",
    "ML": "Biomarker or regulation linked to disease",
    "Mp": "Biomarker of progression",
    "N": "Inhibits",
    "O": "Transport or channel activity",
    "P": "Role in pathogenesis or progression",
    "Pr": "Prevents, suppresses, alleviates, or reduces",
    "Q": "Production by cell population",
    "Ra": "Enhances response or activates",
    "Rg": "Regulation and pathway relationship",
    "Rg+": "Positive regulation and pathway relationship",
    "Sa": "Side effect or adverse event",
    "T": "Treatment or therapy",
    "Te": "Possible therapeutic effect",
    "U": "Mutation or polymorphism alters risk",
    "V+": "Unknown",
    "X": "Possible therapeutic effect",
    "Z": "Enzyme activity",
    "An": "Ancestor disease relationship",
    "As": "Disease association",
    "Chemical-Disease": "Association between chemical and disease",
    "chemical-disease": "Association between chemical and disease",
    "gene-chemical": "Association between gene and chemical",
}


def _normalize_relation(value: object) -> str:
    return str(value).strip()


def get_edge_relation(edge_data: dict) -> str:
    """First present relation key on an edge (matches kgml graphsage_edge._get_edge_relation)."""
    for attr in ("relationship", "predicate", "relationship_type", "label", "edge_type"):
        if attr in edge_data and edge_data[attr] is not None:
            return _normalize_relation_key(edge_data[attr])
    return "UNK"


def _normalize_relation_key(value: object) -> str:
    return str(value).strip()


def get_edge_types(graph: nx.Graph) -> list[str]:
    edge_types: set[str] = set()
    for _, _, data in graph.edges(data=True):
        edge_type = (
            data.get("relationship")
            or data.get("edge_type")
            or data.get("relationship_type")
            or data.get("label")
            or data.get("predicate")
        )
        if edge_type:
            edge_types.add(_normalize_relation(edge_type))
    return sorted(edge_types)


def build_relation_lookup(
    relation_keys: list[str],
) -> tuple[dict[str, int], int]:
    """
    UNK is index 0. Registers lower/upper aliases like kgml _build_relation_matrix.
    Returns (lookup, unk_index=0).
    """
    lookup: dict[str, int] = {"UNK": 0}
    next_idx = 1
    for key in sorted(set(relation_keys)):
        normalized = _normalize_relation_key(key)
        if normalized.upper() == "UNK":
            for alias in {normalized, normalized.lower(), normalized.upper()}:
                lookup[alias] = 0
            continue
        idx = next_idx
        next_idx += 1
        for alias in {normalized, normalized.lower(), normalized.upper()}:
            if alias not in lookup:
                lookup[alias] = idx
    return lookup, 0
