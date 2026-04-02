# Experiments Guide

This repository supports two main experiment families:

1. Generic link prediction on a homogeneous `networkx.Graph` view of the KG.
2. TxGNN-style typed link prediction on a heterogeneous graph.

The default benchmark for link prediction is the node-disjoint split. That is the main setting to use when comparing methods intended to generalize to unseen entities.

## Environment

Create or reuse the repo virtualenv:

```bash
./scripts/setup_hpc.sh
```

Or install locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional semantic relation embeddings for the pickle-only edge-aware runner:

```bash
pip install -e ".[semantic]"
export OPENAI_API_KEY=...
```

## Supported Inputs

### Generic runner: `run_gpu_method`

Path: `python -m kgml_new.scripts.run_gpu_method`

Accepted inputs:

- PrimeKG-style CSV with headers such as `x_name,y_name,relation,x_type,y_type`
- Generic CSV with custom `--source-col`, `--target-col`, `--relation-col`
- Headerless TSV edge lists such as raw `drkg.tsv`
- Pickled `networkx.Graph`

The loader now auto-detects:

- comma vs tab delimiter
- headered vs headerless tables
- DRKG-style node types from identifiers like `Gene::2157`

### Pickle-only runners

These still require a pickled `networkx.Graph`:

- `python -m kgml_new.scripts.run_link_prediction`
- `python -m kgml_new.scripts.run_node_classification`

### TxGNN runner

Path: `python -m kgml_new.scripts.run_txgnn`

Accepted inputs:

- typed CSV/TSV through the generic loader
- pickled `networkx.Graph`

TxGNN requires typed nodes. PrimeKG satisfies this directly through `x_type` and `y_type`. DRKG can work when node types are recoverable from prefixes like `Gene::`, `Compound::`, `Disease::`.

## Main Split Protocols

### `--split-protocol node`

This is the default and recommended setting.

- Nodes are split into train, validation, and test partitions.
- Training edges only connect train nodes.
- Validation edges touch validation nodes but not test nodes.
- Test edges touch test nodes.
- Default negative sampling becomes `type_matched`.

Use this when you want OOD-style evaluation on unseen entities.

### `--split-protocol edge`

Legacy edge-disjoint split.

- Nodes can appear in train and test.
- Only edges are held out.
- Default negative sampling becomes `global`.

Use this only for legacy comparisons or sanity checks.

## Generic Link Prediction Runner

Entry point:

```bash
python -m kgml_new.scripts.run_gpu_method --help
```

Methods:

- `baseline_sage`
- `baseline_gcn`
- `edge_aware_sage`
- `node2vec`
- `link_mlp`

### Common commands

PrimeKG, node-disjoint baseline GraphSAGE:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input kg.csv \
  --split-protocol node \
  --epochs 200 \
  --output results/baseline_sage-node.json
```

PrimeKG, legacy edge split:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input kg.csv \
  --split-protocol edge \
  --epochs 200 \
  --output results/baseline_sage-edge.json
```

DRKG raw TSV, node-disjoint baseline GraphSAGE:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input drkg.tsv \
  --split-protocol node \
  --epochs 200 \
  --output results/drkg-baseline_sage-node.json
```

DRKG smoke test on a subset:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input drkg.tsv \
  --max-edges 1000 \
  --epochs 1 \
  --output /tmp/drkg-smoke.json
```

Edge-aware GraphSAGE:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input kg.csv \
  --split-protocol node \
  --epochs 200 \
  --output results/edge_aware_sage-node.json
```

Node2Vec:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method node2vec \
  --input kg.csv \
  --split-protocol edge \
  --epochs 20 \
  --output results/node2vec-edge.json
```

`node2vec` is not part of the main node-disjoint comparison because unseen nodes do not receive inductive embeddings. The script emits `comparable_under_node_split: false` under node split for that reason.

Link MLP scorer:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method link_mlp \
  --input kg.csv \
  --split-protocol node \
  --epochs 50 \
  --output results/link_mlp-node.json
```

### Important arguments

- `--method`: model family to run
- `--input`: CSV, TSV, or pickle path
- `--input-format {auto,csv,pickle}`: leave as `auto` unless detection is wrong
- `--max-edges`: useful for smoke tests
- `--source-col`, `--target-col`, `--relation-col`: map non-PrimeKG schemas
- `--source-type-col`, `--target-type-col`: typed-node columns when present
- `--split-protocol {node,edge}`: main evaluation choice
- `--negative-sampling-mode {global,type_matched}`: defaults to `type_matched` under node split, `global` under edge split
- `--negatives-per-pos`: number of negatives per positive in validation/test
- `--decoder {dot,mlp}`: link scoring head
- `--shuffle-relations`: relation-label ablation
- `--epochs`
- `--seed`
- `--in-dim`
- `--output`

### Method-specific behavior

`baseline_sage`

- mini-batch neighbor-sampled training
- inductive evaluation compatible with node-disjoint split

`baseline_gcn`

- full-graph training
- slower on large graphs than mini-batch GraphSAGE

`edge_aware_sage`

- relation-aware message passing
- in `run_gpu_method`, relation embeddings are random learned tables initialized in-code
- in `run_link_prediction`, the edge-aware pickle-only path can optionally use OpenAI semantic embeddings

`node2vec`

- transductive embedding baseline
- valid for edge split
- not a main comparable baseline under node-disjoint split

`link_mlp`

- trains a baseline GraphSAGE encoder first
- then trains an MLP scorer on pair features

## Pickled Graph Runner

Entry point:

```bash
python -m kgml_new.scripts.run_link_prediction --help
```

Use this if you already have a `networkx.Graph` pickle and want the older direct runner.

Example:

```bash
.venv/bin/python -m kgml_new.scripts.run_link_prediction \
  --graph data/graph.pkl \
  --model sage \
  --split-protocol node \
  --epochs 20 \
  --out artifacts/embeddings.pt
```

Edge-aware semantic example:

```bash
.venv/bin/python -m kgml_new.scripts.run_link_prediction \
  --graph data/graph.pkl \
  --model edge_sage \
  --semantic \
  --cache cache/relation_emb.pkl \
  --split-protocol node \
  --epochs 20 \
  --out artifacts/edge_sage.pt
```

Important arguments:

- `--graph`: pickled `networkx.Graph`
- `--model {sage,edge_sage}`
- `--semantic / --no-semantic`
- `--cache`: relation embedding cache
- `--split-protocol {node,edge}`
- `--negative-sampling-mode {global,type_matched}`
- `--negatives-per-pos`
- `--decoder {dot,mlp}`
- `--shuffle-relations`
- `--epochs`
- `--seed`
- `--out`

## TxGNN Experiments

Entry point:

```bash
python -m kgml_new.scripts.run_txgnn --help
```

PrimeKG example:

```bash
.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --input kg.csv \
  --relation indication \
  --epochs 20 \
  --output results/txgnn-indication.json
```

When a relation name appears across multiple typed edges, disambiguate it:

```bash
.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --input kg.csv \
  --relation indication \
  --source-node-type drug \
  --target-node-type disease \
  --epochs 20 \
  --output results/txgnn-drug-indication-disease.json
```

Important arguments:

- `--input`, `--input-format`, `--max-edges`
- `--source-col`, `--target-col`, `--relation-col`
- `--source-type-col`, `--target-type-col`
- `--relation`: target relation to predict
- `--source-node-type`, `--target-node-type`: disambiguate typed edge
- `--epochs`
- `--batch-size`
- `--neg-samples`
- `--seed`
- `--in-dim`
- `--output`

TxGNN uses a relation-specific split on the requested edge type and reports validation/test AUC and AP.

## Node Classification

Entry point:

```bash
python -m kgml_new.scripts.run_node_classification --help
```

This path currently expects a pickled `networkx.Graph`.

Example:

```bash
.venv/bin/python -m kgml_new.scripts.run_node_classification \
  --graph data/graph.pkl \
  --method sage \
  --label-attr label \
  --epochs 50 \
  --classifier-epochs 50 \
  --out results/node_cls.json
```

Important arguments:

- `--graph`
- `--method {sage,gcn,edge_sage,node2vec,link_mlp,txgnn}`
- `--label-attr`
- `--target-node-type`
- `--epochs`
- `--classifier-epochs`
- `--seed`
- `--in-dim`
- `--hidden-dim`
- `--embedding-dim`
- `--edge-dim`
- `--batch-size`
- `--out`

## Output Format

Experiment runners write JSON with fields like:

- `method`
- `input_path`
- `split_protocol`
- `num_nodes`
- `num_train_edges`
- `num_val_edges`
- `num_test_edges`
- `history`
- `val_metrics`
- `test_metrics`
- `metrics`

For node-disjoint link prediction, `metrics` is the test-set metrics.

## Recommended Experiment Matrix

For the main link-prediction study, run:

1. `baseline_sage` with `--split-protocol node`
2. `baseline_gcn` with `--split-protocol node`
3. `edge_aware_sage` with `--split-protocol node`
4. `link_mlp` with `--split-protocol node`

Optional baselines:

1. `node2vec` with `--split-protocol edge`
2. relation-shuffle ablation via `--shuffle-relations`
3. decoder ablation via `--decoder mlp`
4. negative-sampling ablation via `--negative-sampling-mode global`

## SLURM

Existing batch scripts:

- `scripts/slurm/run_baseline_sage.slurm`
- `scripts/slurm/run_edge_aware_sage.slurm`
- `scripts/slurm/run_baseline_gcn.slurm`
- `scripts/slurm/run_node2vec.slurm`
- `scripts/slurm/run_link_mlp.slurm`
- `scripts/slurm/run_txgnn.slurm`

Submit with:

```bash
sbatch scripts/slurm/run_baseline_sage.slurm
```

Adjust paths, partitions, memory, and output locations for your cluster before submitting.

## Practical Notes

- Use `--max-edges` first for smoke tests before launching full runs.
- `run_gpu_method` is the easiest entrypoint for raw PrimeKG CSV and raw DRKG TSV.
- `run_link_prediction` is better when you already have a pickled graph and want semantic relation embeddings.
- Under node split, a method must support inductive embeddings for unseen nodes to be part of the main comparison.
- DRKG can be extremely large. Start with subsets to validate runtime and memory on your machine.
