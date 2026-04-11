from __future__ import annotations

import argparse
import bz2
import json
import pickle
from pathlib import Path

import networkx as nx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Hetionet v1 JSON(.bz2) into kgml_new-ready formats "
            "(headerless TSV edge list and/or pickled networkx graph)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to hetionet-v1.0.json or hetionet-v1.0.json.bz2",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=None,
        help="Write headerless TSV edges as: source relation target",
    )
    parser.add_argument(
        "--output-pickle",
        type=Path,
        default=None,
        help="Write networkx.Graph pickle for direct --input-format pickle usage",
    )
    parser.add_argument(
        "--encode-direction-in-relation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append direction to relation label when not 'both' to preserve directed "
            "semantics in undirected training graphs."
        ),
    )
    return parser.parse_args()


def _load_hetionet_json(path: Path) -> dict:
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload, got {type(payload)}")
    if "nodes" not in payload or "edges" not in payload:
        raise ValueError("Hetionet payload missing required keys: nodes and edges")
    return payload


def _node_name(kind: str, identifier: object) -> str:
    return f"{kind}::{identifier}"


def _edge_relation(kind: str, direction: str, *, encode_direction: bool) -> str:
    relation = str(kind).strip()
    direction_value = str(direction).strip().lower()
    if not encode_direction or direction_value in {"", "both"}:
        return relation
    return f"{relation}::{direction_value}"


def _build_node_map(nodes: list[dict]) -> dict[tuple[str, str], str]:
    node_map: dict[tuple[str, str], str] = {}
    for node in nodes:
        kind = str(node["kind"])
        identifier = str(node["identifier"])
        name = _node_name(kind, identifier)
        node_map[(kind, identifier)] = name
    return node_map


def write_tsv(
    *,
    payload: dict,
    output_path: Path,
    encode_direction_in_relation: bool,
) -> tuple[int, int]:
    node_map = _build_node_map(payload["nodes"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edge_count = 0
    relation_labels: set[str] = set()
    with output_path.open("w", encoding="utf-8") as handle:
        for edge in payload["edges"]:
            src_kind, src_identifier = edge["source_id"]
            dst_kind, dst_identifier = edge["target_id"]

            src = node_map[(str(src_kind), str(src_identifier))]
            dst = node_map[(str(dst_kind), str(dst_identifier))]
            relation = _edge_relation(
                str(edge["kind"]),
                str(edge.get("direction", "both")),
                encode_direction=encode_direction_in_relation,
            )
            handle.write(f"{src}\t{relation}\t{dst}\n")
            edge_count += 1
            relation_labels.add(relation)
    return edge_count, len(relation_labels)


def write_pickle(
    *,
    payload: dict,
    output_path: Path,
    encode_direction_in_relation: bool,
) -> tuple[int, int]:
    graph = nx.Graph()

    for node in payload["nodes"]:
        kind = str(node["kind"])
        identifier = str(node["identifier"])
        node_name = _node_name(kind, identifier)
        graph.add_node(
            node_name,
            node_type=kind.lower(),
            node_id=identifier,
            node_label=str(node.get("name", node_name)),
        )

    relation_labels: set[str] = set()
    for edge in payload["edges"]:
        src_kind, src_identifier = edge["source_id"]
        dst_kind, dst_identifier = edge["target_id"]
        src = _node_name(str(src_kind), str(src_identifier))
        dst = _node_name(str(dst_kind), str(dst_identifier))
        relation = _edge_relation(
            str(edge["kind"]),
            str(edge.get("direction", "both")),
            encode_direction=encode_direction_in_relation,
        )
        graph.add_edge(src, dst, relationship=relation)
        relation_labels.add(relation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(graph, handle)
    return int(graph.number_of_edges()), len(relation_labels)


def main() -> None:
    args = parse_args()
    if args.output_tsv is None and args.output_pickle is None:
        raise SystemExit("Set at least one of --output-tsv or --output-pickle")

    payload = _load_hetionet_json(args.input)
    num_nodes = len(payload["nodes"])
    num_edges = len(payload["edges"])
    print(f"Loaded Hetionet payload: nodes={num_nodes} edges={num_edges}")

    if args.output_tsv is not None:
        tsv_edges, tsv_relations = write_tsv(
            payload=payload,
            output_path=args.output_tsv,
            encode_direction_in_relation=args.encode_direction_in_relation,
        )
        print(
            f"Wrote TSV: {args.output_tsv} "
            f"(edges={tsv_edges}, unique_relations={tsv_relations})"
        )

    if args.output_pickle is not None:
        pkl_edges, pkl_relations = write_pickle(
            payload=payload,
            output_path=args.output_pickle,
            encode_direction_in_relation=args.encode_direction_in_relation,
        )
        print(
            f"Wrote pickle graph: {args.output_pickle} "
            f"(edges={pkl_edges}, unique_relations={pkl_relations})"
        )


if __name__ == "__main__":
    main()
