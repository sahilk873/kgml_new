from kgml_new.data.graph import networkx_to_data
from kgml_new.data.hetero import networkx_to_heterodata
from kgml_new.data.relations import (
    RELATION_DESCRIPTIONS,
    build_relation_lookup,
    get_edge_relation,
    get_edge_types,
)

__all__ = [
    "RELATION_DESCRIPTIONS",
    "build_relation_lookup",
    "get_edge_relation",
    "get_edge_types",
    "networkx_to_data",
    "networkx_to_heterodata",
]
