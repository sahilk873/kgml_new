"""Export full-graph HGT node embeddings from a saved encoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kgml_new.data.loaders import GraphCSVSpec, PRIMEKG_CSV_SPEC, load_graph_csv, load_pickled_graph
from kgml_new.data.hetero import networkx_to_heterodata
from kgml_new.models.hgt import HGTLinkPredictor

from kgml_new.scripts.run_hgt import (
    _link_split_for_hgt,
    _resolve_target_edge_type,
)


def _csv_spec_from_export_args(args: argparse.Namespace) -> GraphCSVSpec:
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


def _load_graph_export(args: argparse.Namespace):
    input_format = args.input_format
    if input_format == "auto":
        suffix = args.input_path.suffix.lower()
        input_format = "pickle" if suffix in {".pkl", ".pickle"} else "csv"
    if input_format == "pickle":
        return load_pickled_graph(args.input_path)
    return load_graph_csv(args.input_path, spec=_csv_spec_from_export_args(args), max_edges=args.max_edges)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load HGT encoder artifact, rebuild train-graph HeteroData, export node embeddings."
    )
    p.add_argument("--encoder", type=Path, required=True, help="Path to *.encoder.pt from run_hgt.")
    p.add_argument("--output", type=Path, required=True, help="Output .pt path (dict payload).")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda or cpu (default: cuda if available).",
    )
    p.add_argument(
        "--node-type",
        type=str,
        default=None,
        help="Emit only this heterogeneous node type (required if multiple types exist).",
    )
    p.add_argument("--input", "--csv", dest="input_path", type=Path, default=Path("data/kg.csv"))
    p.add_argument("--input-format", choices=("auto", "csv", "pickle"), default="auto")
    p.add_argument("--max-edges", type=int, default=None)
    p.add_argument("--source-col", type=str, default=PRIMEKG_CSV_SPEC.source_col)
    p.add_argument("--target-col", type=str, default=PRIMEKG_CSV_SPEC.target_col)
    p.add_argument("--relation-col", type=str, default=PRIMEKG_CSV_SPEC.relation_col)
    p.add_argument("--source-type-col", type=str, default=PRIMEKG_CSV_SPEC.source_type_col)
    p.add_argument("--target-type-col", type=str, default=PRIMEKG_CSV_SPEC.target_type_col)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in-dim", type=int, default=64, help="Fallback if missing from checkpoint.")
    p.add_argument("--num-layers", type=int, default=2, help="Fallback if missing from checkpoint.")
    p.add_argument("--num-heads", type=int, default=4, help="Fallback if missing from checkpoint.")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--split-protocol", choices=("edge", "node"), default="edge")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument(
        "--negative-sampling-mode",
        choices=("global", "type_matched"),
        default="global",
    )
    p.add_argument("--relation", type=str, default="indication")
    p.add_argument("--source-node-type", type=str, default=None)
    p.add_argument("--target-node-type", type=str, default=None)
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    ckpt_path = Path(args.encoder).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"Encoder checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise SystemExit("Checkpoint missing state_dict.")

    in_dim = int(ckpt.get("in_dim", args.in_dim))
    num_layers = int(ckpt.get("num_layers", args.num_layers))
    num_heads = int(ckpt.get("num_heads", args.num_heads))

    torch.manual_seed(args.seed)
    graph = _load_graph_export(args)
    data = networkx_to_heterodata(graph, in_dim=in_dim, seed=args.seed, add_reverse_edges=True)
    edge_type_list = ckpt.get("edge_type")
    if isinstance(edge_type_list, list) and len(edge_type_list) == 3:
        edge_type = (edge_type_list[0], edge_type_list[1], edge_type_list[2])
    else:
        edge_type = _resolve_target_edge_type(args, list(data.edge_types))

    train_pos, _val_pos, _val_neg, _test_pos, _test_neg, _bipartite_ns = _link_split_for_hgt(
        args=args,
        data=data,
        edge_type=edge_type,
        graph=graph,
    )
    data[edge_type].edge_index = train_pos
    rev_edge_type = (edge_type[2], f"rev_{edge_type[1]}", edge_type[0])
    if rev_edge_type in data.edge_types:
        data[rev_edge_type].edge_index = train_pos.flip(0)

    model = HGTLinkPredictor(
        metadata=data.metadata(),
        in_channels=in_dim,
        hidden_channels=in_dim,
        out_channels=in_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=float(args.dropout),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        z_dict = model.encode(data.to(device))

    node_type = args.node_type
    if node_type is None:
        keys = list(z_dict.keys())
        if len(keys) == 1:
            node_type = keys[0]
        else:
            raise SystemExit(
                f"Multiple node types {keys}: pass --node-type with the tensor to export."
            )

    emb = z_dict[node_type].detach().cpu()
    payload = {
        "artifact_type": "hgt_export_embeddings",
        "encoder_checkpoint": str(ckpt_path),
        "node_type": node_type,
        "embeddings": emb,
        "edge_type": list(edge_type),
        "split_protocol": args.split_protocol,
        "input_path": str(Path(args.input_path).resolve()),
        "num_nodes": int(emb.size(0)),
        "embedding_dim": int(emb.size(1)),
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    summary = {"ok": True, "output": str(out_path), "shape": list(emb.shape)}
    print(json.dumps(summary, indent=2))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
