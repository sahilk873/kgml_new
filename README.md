# kgml_new

Standalone **PyTorch Geometric** implementation of:

- **Baseline GraphSAGE** (`SAGEConv`, mean aggregation, L2-normalized outputs)
- **Edge-aware semantic GraphSAGE**: neighbor messages use `concat(node_features, relation_embedding)` per edge, matching the kgml `TensorizedEdgeAwareMeanAggregator` design
- **BaselineGCN** for comparison
- **Node2Vec** embeddings
- **TxGNN-style hetero model**: relation-aware heterogeneous GNN + DistMult decoder + disease prototype augmentation

GraphSAGE training defaults to mini-batch neighbor sampling via `LinkNeighborLoader`, with loader-level negative edge sampling for link prediction. For link prediction, the default evaluation now uses a node-disjoint split so held-out entities are unseen during training edges, which makes the benchmark closer to OOD generalization. Full-graph training remains available as an explicit fallback path.

No Kedro. Intended for graphs built from **PrimeKG** or any **NetworkX** graph with relation attributes on edges.

## Install

```bash
cd kgml_new
pip install -e ".[dev]"           # core + pytest
pip install -e ".[semantic]"        # optional: OpenAI relation embeddings
```

Set `OPENAI_API_KEY` when using `--semantic` / OpenAI path.
For HPC or fresh cluster installs, prefer `./scripts/setup_hpc.sh` because it also installs the matching PyTorch Geometric backends.

## CLI

```bash
python -m kgml_new.scripts.run_link_prediction --graph /path/to/graph.pkl --model sage --epochs 20 --out emb.pt
python -m kgml_new.scripts.run_link_prediction --graph /path/to/graph.pkl --model edge_sage --no-semantic --epochs 20
python -m kgml_new.scripts.run_gpu_method --method baseline_sage --input kg.csv --source-col x_name --target-col y_name --relation-col relation --epochs 3 --output results.json
python -m kgml_new.scripts.run_txgnn --input kg.csv --source-col x_name --target-col y_name --relation-col relation --relation indication --epochs 20 --output txgnn.json
```

`run_gpu_method` and `run_txgnn` now accept either generic CSV input or a pickled NetworkX graph via `--input-format {auto,csv,pickle}`. For non-PrimeKG CSVs, point `--source-col`, `--target-col`, `--relation-col`, and optional type columns at the right schema instead of changing code.
For full experiment instructions, including the node-disjoint protocol, DRKG support, TxGNN runs, and CLI options, see `EXPERIMENTS.md`.

## HPC Setup

```bash
./scripts/setup_hpc.sh
./scripts/prepare_data.sh
```

## PrimeKG

Export or convert your graph to **NetworkX** with one attribute per edge for the relation (see `get_edge_relation` in `kgml_new.data.relations`: `relationship`, `predicate`, `relationship_type`, `label`, or `edge_type`). Map PrimeKG predicates to short strings; optional glosses live in `RELATION_DESCRIPTIONS` (extend as needed).

## License

Same as parent repository unless noted otherwise.
