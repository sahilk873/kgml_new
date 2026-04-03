# kgml_new

Standalone **PyTorch Geometric** implementation of:

- **Baseline GraphSAGE** (`SAGEConv`, mean aggregation, L2-normalized outputs)
- **Edge-aware semantic GraphSAGE** with three relation-control modes:
  - `concat`: current baseline, matching the kgml `TensorizedEdgeAwareMeanAggregator` style
  - `gated`: relation embedding gates the learned neighbor message
  - `basis_mixture`: relation embedding mixes over a small set of learned basis message operators
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

Semantic relation embeddings now use `relation_glossary.tsv` for DRKG-style sourced predicates and fall back to the raw relation string otherwise. PrimeKG-style labels remain raw-string prompts unless a separate PrimeKG glossary is added.

## CLI

```bash
python -m kgml_new.scripts.run_link_prediction --graph /path/to/graph.pkl --model sage --epochs 20 --out emb.pt
python -m kgml_new.scripts.run_link_prediction --graph /path/to/graph.pkl --model edge_sage --edge-relation-mode gated --no-semantic --epochs 20
python -m kgml_new.scripts.run_gpu_method --method baseline_sage --input kg.csv --source-col x_name --target-col y_name --relation-col relation --epochs 3 --output results.json
python -m kgml_new.scripts.run_gpu_method --method edge_aware_sage --input drkg.tsv --semantic --semantic-cache cache/drkg-relations.pt --edge-relation-mode concat --epochs 20 --output drkg-edge-aware.json
python -m kgml_new.scripts.run_gpu_method --method edge_aware_sage --input drkg.tsv --semantic --semantic-cache cache/drkg-relations.pt --edge-relation-mode gated --epochs 20 --output drkg-edge-aware-gated.json
python -m kgml_new.scripts.run_gpu_method --method edge_aware_sage --input drkg.tsv --semantic --semantic-cache cache/drkg-relations.pt --edge-relation-mode basis_mixture --num-relation-bases 4 --epochs 20 --output drkg-edge-aware-basis.json
python -m kgml_new.scripts.generate_relation_embeddings --input drkg.tsv --output cache/drkg-relations.pt --semantic
python -m kgml_new.scripts.run_txgnn --input kg.csv --source-col x_name --target-col y_name --relation-col relation --relation indication --epochs 20 --output txgnn.json
python -m kgml_new.scripts.run_node_classification --graph /path/to/graph.pkl --method edge_sage --semantic --semantic-cache cache/relations.pt --edge-relation-mode basis_mixture --num-relation-bases 4 --out node-cls.json
```

`run_gpu_method` and `run_txgnn` now accept either generic CSV input or a pickled NetworkX graph via `--input-format {auto,csv,pickle}`. For non-PrimeKG CSVs, point `--source-col`, `--target-col`, `--relation-col`, and optional type columns at the right schema instead of changing code.
For full experiment instructions, including the node-disjoint protocol, DRKG support, TxGNN runs, and CLI options, see `EXPERIMENTS.md`.

## Edge Relation Modes

All edge-aware GraphSAGE paths now support:

- `--edge-relation-mode concat`: baseline path, projected relation embedding is concatenated with `x_j`
- `--edge-relation-mode gated`: projected relation embedding produces a sigmoid gate over the learned neighbor message
- `--edge-relation-mode basis_mixture`: projected relation embedding produces mixture weights over `K` learned basis message transforms

The raw relation table remains frozen by default in all three modes. Semantic and random relation tables both reuse the same cache and lookup pipeline. The projected relation embedding is still trainable through the shared `relation_projection`.

`--num-relation-bases K` only matters for `basis_mixture`.

## Semantic Relation Cache

You can precompute the DRKG glossary-first relation embedding cache and reuse it across runs:

```bash
.venv/bin/python -m kgml_new.scripts.generate_relation_embeddings \
  --input drkg.tsv \
  --output cache/drkg-relations.pt \
  --semantic
```

Then point the edge-aware runner at the same cache:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input drkg.tsv \
  --semantic \
  --semantic-cache cache/drkg-relations.pt \
  --edge-relation-mode concat \
  --epochs 200 \
  --output results/drkg-edge-aware.json
```

For DRKG-style sourced predicates, `relation_glossary.tsv` supplies the `Data-source`, `Connected entity-types`, `Interaction-type`, and `Description` fields for the OpenAI prompt. PrimeKG-style labels and any relation without a DRKG-style glossary match are embedded from the raw relation string. The cached semantic vectors keep their full OpenAI width, and the edge-aware model learns a projection into the configured message-space `edge_dim`.
Legacy semantic caches saved before the full-width change are automatically regenerated when reused.

If you already have a full DRKG relation cache such as `cache/drkg-relations.pt`, it can be reused for smaller DRKG subsets as long as the relation names in the subset are covered by the cache. Slice-specific caches are only needed when you want the experiment artifact to be fully self-contained.

## DRKG Slice Caveat

`drkg.tsv` is ordered by relation block. A plain prefix slice like `--max-edges 25000` can contain only one relation type, which makes semantic-vs-random comparisons misleading.

For relation-diverse DRKG subsets, prefer one of:

- shuffled DRKG file, then take the first `N` rows
- random row sampling
- relation-stratified sampling
- a later contiguous window with higher relation diversity

For example, a reproducible shuffle can be written once and then reused:

```bash
python3 - <<'PY'
from pathlib import Path
import random
seed = 42
src = Path('drkg.tsv')
out = Path('data/drkg-shuffled-seed42.tsv')
random.seed(seed)
lines = src.read_text().splitlines()
random.shuffle(lines)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\\n'.join(lines) + '\\n')
PY
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
