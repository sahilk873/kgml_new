# kgml_new

Standalone **PyTorch Geometric** implementation of:

- **Baseline GraphSAGE** (`SAGEConv`, mean aggregation, L2-normalized outputs)
- **Edge-aware semantic GraphSAGE**: neighbor messages use `concat(node_features, relation_embedding)` per edge, matching the kgml `TensorizedEdgeAwareMeanAggregator` design
- **BaselineGCN** for comparison
- **Node2Vec** embeddings
- **TxGNN-style hetero model**: relation-aware heterogeneous GNN + DistMult decoder + disease prototype augmentation

No Kedro. Intended for graphs built from **PrimeKG** or any **NetworkX** graph with relation attributes on edges.

## Install

```bash
cd kgml_new
pip install -e ".[dev]"           # core + pytest
pip install -e ".[semantic]"        # optional: OpenAI relation embeddings
```

Set `OPENAI_API_KEY` when using `--semantic` / OpenAI path.

## CLI

```bash
python -m kgml_new.scripts.run_link_prediction --graph /path/to/graph.pkl --model sage --epochs 20 --out emb.pt
python -m kgml_new.scripts.run_link_prediction --graph /path/to/graph.pkl --model edge_sage --no-semantic --epochs 20
python -m kgml_new.scripts.run_txgnn --csv kg.csv --max-edges 10000 --relation indication --epochs 20 --output txgnn.json
```

## HPC Setup

```bash
./scripts/setup_hpc.sh
./scripts/prepare_data.sh
```

## PrimeKG

Export or convert your graph to **NetworkX** with one attribute per edge for the relation (see `get_edge_relation` in `kgml_new.data.relations`: `relationship`, `predicate`, `relationship_type`, `label`, or `edge_type`). Map PrimeKG predicates to short strings; optional glosses live in `RELATION_DESCRIPTIONS` (extend as needed).

## License

Same as parent repository unless noted otherwise.
