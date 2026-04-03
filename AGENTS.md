# AGENTS.md

This file is the operating guide for agents working in `kgml_new`. It is intended to be the single place to understand what this repo does, how experiments are run, what inputs are supported, how semantic relation embeddings behave, what caches exist, and what caveats matter before launching or interpreting runs.

## Repo Purpose

`kgml_new` is a standalone PyTorch Geometric implementation for knowledge-graph experiments. The main focus is link prediction, with additional support for node classification and a TxGNN-style heterogeneous model.

The primary model families in this repository are:

- Baseline GraphSAGE
- Edge-aware GraphSAGE
- Baseline GCN
- Node2Vec
- Link MLP on top of learned node embeddings
- Edge-aware Link MLP
- TxGNN-style heterogeneous GNN

The repo is intended to work with:

- PrimeKG-style CSV data
- DRKG-style headerless TSV edge lists
- generic CSV or TSV edge lists with configurable columns
- pickled `networkx.Graph` objects

## Ground Truth Files To Trust

When in doubt, trust these files first:

- [README.md](/Users/sahilkapadia/development/kgml/kgml_new/README.md)
- [EXPERIMENTS.md](/Users/sahilkapadia/development/kgml/kgml_new/EXPERIMENTS.md)
- [src/kgml_new/scripts/run_gpu_method.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_gpu_method.py)
- [src/kgml_new/scripts/run_txgnn.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_txgnn.py)
- [src/kgml_new/scripts/run_link_prediction.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_link_prediction.py)
- [src/kgml_new/scripts/run_node_classification.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_node_classification.py)
- [src/kgml_new/embeddings/semantic.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/embeddings/semantic.py)
- [src/kgml_new/config.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/config.py)
- [src/kgml_new/data/loaders.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/data/loaders.py)
- [src/kgml_new/data/relations.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/data/relations.py)

## Environment And Setup

Python packaging is defined in [pyproject.toml](/Users/sahilkapadia/development/kgml/kgml_new/pyproject.toml).

Core dependencies:

- `networkx>=3.0`
- `numpy>=1.24`
- `scikit-learn>=1.3`
- `torch>=2.0`
- `torch-geometric>=2.4`
- `pandas>=2.0`

Optional semantic dependencies:

- `openai>=1.0`
- `python-dotenv>=1.0`

Common setup flows:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -e ".[semantic]"
```

For cluster or HPC installs:

```bash
./scripts/setup_hpc.sh
```

If semantic relation embeddings are used, `OPENAI_API_KEY` must be present. The repo loads `.env` automatically via `python-dotenv`.

## Key Local Paths

Important files and directories in this repo:

- [kg.csv](/Users/sahilkapadia/development/kgml/kgml_new/kg.csv): PrimeKG-style CSV
- [drkg.tsv](/Users/sahilkapadia/development/kgml/kgml_new/drkg.tsv): DRKG edge list TSV
- [relation_glossary.tsv](/Users/sahilkapadia/development/kgml/kgml_new/relation_glossary.tsv): glossary used for DRKG-style sourced predicates
- [cache](/Users/sahilkapadia/development/kgml/kgml_new/cache): relation embedding caches
- [results](/Users/sahilkapadia/development/kgml/kgml_new/results): experiment outputs if created
- [scripts/slurm](/Users/sahilkapadia/development/kgml/kgml_new/scripts/slurm): SLURM launch scripts

## Main Experiment Entry Points

### 1. `run_gpu_method`

Primary entry point for generic link prediction experiments:

- [src/kgml_new/scripts/run_gpu_method.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_gpu_method.py)

Use this for almost all standard method comparisons.

Supported methods:

- `baseline_sage`
- `baseline_gcn`
- `edge_aware_sage`
- `node2vec`
- `link_mlp`
- `edge_aware_link_mlp`

For edge-aware methods, the relation mechanism is now selected separately with:

- `--edge-relation-mode concat`
- `--edge-relation-mode gated`
- `--edge-relation-mode basis_mixture`

Canonical examples:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input kg.csv \
  --split-protocol node \
  --epochs 200 \
  --output results/baseline_sage-node.json
```

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input drkg.tsv \
  --split-protocol node \
  --semantic \
  --semantic-cache cache/drkg-relations.pt \
  --edge-relation-mode concat \
  --epochs 200 \
  --output results/drkg-edge-aware-sage-node.json
```

`run_gpu_method` is the main end-to-end link prediction runner. It:

1. loads a graph from CSV, TSV, or pickle
2. builds the link prediction split and negative edges
3. trains one chosen method
4. evaluates validation and test performance
5. writes a JSON artifact with metrics and run metadata

### 2. `run_txgnn`

Typed heterogeneous KG runner:

- [src/kgml_new/scripts/run_txgnn.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_txgnn.py)

Use this when typed nodes and typed edges matter and you want the TxGNN-style setup.

Example:

```bash
.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --input kg.csv \
  --relation indication \
  --epochs 20 \
  --output results/txgnn.json
```

### 3. `run_link_prediction`

Legacy pickle-only link prediction runner:

- [src/kgml_new/scripts/run_link_prediction.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_link_prediction.py)

Only accepts pickled `networkx.Graph` input.

### 4. `run_node_classification`

Node classification runner:

- [src/kgml_new/scripts/run_node_classification.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/run_node_classification.py)

Only accepts pickled `networkx.Graph` input.

For `edge_sage`, node classification now shares the same semantic/random/cache relation-table pipeline as link prediction, including `--edge-relation-mode` and `--num-relation-bases`.

### 5. `generate_relation_embeddings`

Precompute semantic or random relation caches:

- [src/kgml_new/scripts/generate_relation_embeddings.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/scripts/generate_relation_embeddings.py)

This is important for reproducible edge-aware runs and for avoiding repeated OpenAI embedding calls.

Example:

```bash
.venv/bin/python -m kgml_new.scripts.generate_relation_embeddings \
  --input drkg.tsv \
  --output cache/drkg-relations.pt \
  --semantic
```

## Input Formats And How Loading Works

The graph-loading logic lives in [src/kgml_new/data/loaders.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/data/loaders.py).

### Supported input types

- PrimeKG-style CSV with headers
- generic CSV or TSV with custom column names
- headerless TSV edge lists like DRKG
- pickled `networkx.Graph`

### Auto-detection behavior

The loader will infer:

- delimiter: tab if the first line has tabs and not commas, otherwise comma
- whether a header is present
- node types from DRKG-style identifiers like `Gene::2157` if explicit type columns are absent

### PrimeKG schema defaults

The built-in PrimeKG CSV spec uses:

- source column: `x_name`
- target column: `y_name`
- relation column: `relation`
- source type column: `x_type`
- target type column: `y_type`
- source id column: `x_id`
- target id column: `y_id`

### Headerless DRKG behavior

Headerless tables are interpreted as:

- column 0: source
- column 1: relation
- column 2: target

Any extra columns are ignored by default.

### `max_edges` caveat

`--max-edges` slices by taking the first `N` rows after loading the table. It does not stratify or shuffle.

This matters a lot for DRKG because the file is ordered by relation blocks. Taking the first `N` rows may produce a very low-diversity or even single-relation slice.

## Split Protocols

The main evaluation split choices are:

- `--split-protocol node`
- `--split-protocol edge`

### `node` split

Recommended for main experiments.

Behavior:

- nodes are split into train, validation, and test partitions
- training edges only connect train nodes
- validation edges touch validation nodes but not test nodes
- test edges touch test nodes
- default negative sampling switches to `type_matched`

Interpret this as the inductive or OOD-style benchmark.

### `edge` split

Legacy comparison mode.

Behavior:

- nodes may appear in both train and test
- only edges are held out
- default negative sampling switches to `global`

Use only if you explicitly want the older transductive-style benchmark or are reproducing legacy comparisons.

## Negative Sampling

Supported modes:

- `global`
- `type_matched`

Defaults:

- node split: `type_matched`
- edge split: `global`

There is also a `--negatives-per-pos` parameter controlling the number of negatives per positive at validation and test time.

## Decoders

Supported link scoring decoders:

- `dot`
- `mlp`

`dot` is the standard embedding dot-product path.

`mlp` trains an additional scorer using pair features. For `run_gpu_method`, this is exposed via `--decoder mlp`.

## Method Families And What They Mean

### `baseline_sage`

- mini-batch GraphSAGE
- inductive
- main baseline for node-disjoint experiments

### `baseline_gcn`

- full-graph GCN
- usually slower on large graphs than mini-batch GraphSAGE

### `edge_aware_sage`

- relation-aware GraphSAGE
- per-edge message uses relation information
- supports semantic relation embeddings or random relation embeddings

### `node2vec`

- transductive embedding baseline
- not a main comparable method under node split
- only meaningfully comparable under edge split

### `link_mlp`

- GraphSAGE encoder plus learned MLP scorer

### `edge_aware_link_mlp`

- edge-aware GraphSAGE encoder plus learned MLP scorer

### `txgnn`

- typed heterogeneous GNN
- relation-aware heterogeneous model with a DistMult-style decoding path and prototype augmentation

## Semantic Relation Embeddings

Semantic logic lives in [src/kgml_new/embeddings/semantic.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/embeddings/semantic.py).

### Default model

Async OpenAI embeddings call:

- model: `text-embedding-3-small`

### Prompt construction

For DRKG-style sourced predicates containing `::`, the system tries to use [relation_glossary.tsv](/Users/sahilkapadia/development/kgml/kgml_new/relation_glossary.tsv).

Prompt fields may include:

- relation name
- data source
- connected entity types
- interaction type
- description

If a glossary entry is missing, it falls back to a raw-string prompt like:

- `Relationship type: <relation>`

For PrimeKG-style labels such as `drug_protein`, there is no DRKG glossary match and the relation is embedded from the raw string unless a dedicated PrimeKG glossary is added.

### Cache format

Caches are written with:

- `format_version`
- `use_openai`
- `embedding_dim`
- `embeddings`

Current semantic caches keep the full OpenAI embedding width. The edge-aware model learns a projection into message-space `edge_dim`.

### Legacy cache behavior

Older semantic caches may have been stored at message-space width instead of full OpenAI width. The loader detects that and regenerates them automatically when needed.

### Strictness

`--strict-semantic` makes OpenAI setup failures fatal.

Without `--strict-semantic`, semantic failures fall back to random relation embeddings.

## Edge-Aware Model Details That Matter For Interpretation

The implementation is in [src/kgml_new/models/edge_aware_sage.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/models/edge_aware_sage.py).

Important details:

- `relation_table` is stored as an `nn.Parameter(..., requires_grad=False)`
- the raw relation embeddings are frozen
- if the relation input width differs from `edge_dim`, the model learns a projection layer
- there are now three edge-aware relation-control modes:
  - `concat`: projected relation embedding is concatenated with `x_j`
  - `gated`: projected relation embedding produces a sigmoid gate over the learned neighbor message
  - `basis_mixture`: projected relation embedding produces mixture weights over `K` shared basis message transforms
- therefore a run with semantic relation embeddings is usually:
  - frozen semantic table
  - learned projection into edge message space
- a run with random relation embeddings is usually:
  - frozen random table
  - learned projection into edge message space

This means "semantic vs random" is not the same as "fully frozen relation features end-to-end". The projection is trainable unless explicitly changed in code.

## Default Training Hyperparameters

Defaults live in [src/kgml_new/config.py](/Users/sahilkapadia/development/kgml/kgml_new/src/kgml_new/config.py).

### `TrainConfig`

- `in_dim = 64`
- `edge_dim = 32`
- `hidden_dim = 32`
- `out_dim = 64`
- `num_layers = 2`
- `epochs = 100`
- `neg_samples = 5`
- `learning_rate = 1e-3`
- `batch_size = 256`
- `num_neighbors = [10, 10]`
- `seed = 42`
- `dropout = 0.0`
- `concat = True`

### `Node2VecConfig`

- `embedding_dim = 64`
- `walk_length = 10`
- `context_size = 10`
- `walks_per_node = 1`
- `num_negative_samples = 5`
- `epochs = 20`
- `batch_size = 512`
- `learning_rate = 0.01`

### `LinkMLPConfig`

- hidden dims: `(256, 128)`
- dropout: `0.1`
- epochs: `50`
- batch size: `512`
- learning rate: `1e-3`

### `NodeClassificationConfig`

- `in_dim = 64`
- `edge_dim = 32`
- `hidden_dim = 64`
- `embedding_dim = 64`
- `epochs = 50`
- `classifier_epochs = 50`
- `batch_size = 256`

## Datasets In This Repo

### `kg.csv`

PrimeKG-style CSV. Use this for PrimeKG experiments and typed-node experiments that fit the PrimeKG schema.

### `drkg.tsv`

DRKG edge list TSV. Important caveat: this file is relation-block ordered, not shuffled.

This means:

- `--max-edges 25000` does not produce a relation-diverse sample
- taking the first `N` rows can accidentally test only one predicate

### `entity2src.tsv`

Auxiliary file in the repo root. It is not used directly by the main experiment entrypoints shown above, but it may be useful for dataset understanding.

## Concrete DRKG Ordering Caveat Observed In This Repo

These facts were confirmed locally and are important enough to record here:

- first 25k rows of `drkg.tsv` contain exactly 1 relation
- first 50k rows still contain exactly 1 relation
- the first 25k rows are all `bioarx::HumGenHumGen:Gene:Gene`
- within the first 200k rows, a 25k-row contiguous window from rows `60230` to `85229` contains 16 distinct relations

Implication:

- prefix slicing with `--max-edges` on DRKG is not a good semantic-ablation design if the goal is to test relation semantics

Better options:

- choose a later contiguous window with more relation diversity
- sample 25k rows uniformly at random
- stratify by relation
- shuffle the file once, then take the first 25k rows

A reproducible shuffle file is now present in this repo:

- [data/drkg-shuffled-seed42.tsv](/Users/sahilkapadia/development/kgml/kgml_new/data/drkg-shuffled-seed42.tsv)

## Existing Caches And Their Meanings

At the time this file was written, the repo has used these cache names:

- [cache/drkg-relations.pt](/Users/sahilkapadia/development/kgml/kgml_new/cache/drkg-relations.pt): full DRKG relation cache
- [cache/primekg-relations.pt](/Users/sahilkapadia/development/kgml/kgml_new/cache/primekg-relations.pt): PrimeKG relation cache
- [cache/drkg-relations-5k.pt](/Users/sahilkapadia/development/kgml/kgml_new/cache/drkg-relations-5k.pt): first-5k-row DRKG relation cache
- [cache/drkg-relations-25k-semantic.pt](/Users/sahilkapadia/development/kgml/kgml_new/cache/drkg-relations-25k-semantic.pt): semantic cache used for the 25k-row edge-aware experiments
- [cache/drkg-shuffled-seed42-relations-25k-semantic.pt](/Users/sahilkapadia/development/kgml/kgml_new/cache/drkg-shuffled-seed42-relations-25k-semantic.pt): semantic cache used for the shuffled 25k-row DRKG experiment

Do not assume a cache name implies good relation diversity. Check the underlying slice logic.

## Existing Results Worth Knowing

Recent local experiments used:

- 25k-row DRKG prefix slice
- `edge_aware_sage`
- `split-protocol node`
- semantic cache versus `--no-semantic`

At 200 epochs, on that specific low-diversity slice:

- random frozen relation table + learned projection slightly outperformed semantic frozen relation table + learned projection on ROC-AUC and Hits@K
- AP was roughly tied

Interpretation caveat:

- because the 25k-row prefix slice had only one relation type, this is not a strong test of relation semantics

Recent shuffled-slice experiments also used:

- first 25k rows of [data/drkg-shuffled-seed42.tsv](/Users/sahilkapadia/development/kgml/kgml_new/data/drkg-shuffled-seed42.tsv)
- 99 relation types in that slice
- `edge_aware_sage`
- `split-protocol node`
- semantic cache versus `--no-semantic`

At 200 epochs on the shuffled 25k slice:

- semantic and random were very close
- random was slightly better on ROC-AUC, AP, Hits@1, and Hits@3
- semantic was slightly better on Hits@10

This is a much more meaningful semantic-vs-random comparison than the original single-relation prefix slice.

## Standard Commands

### PrimeKG baseline GraphSAGE

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input kg.csv \
  --split-protocol node \
  --epochs 200 \
  --output results/primekg-baseline-sage-node.json
```

### PrimeKG edge-aware semantic GraphSAGE

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input kg.csv \
  --split-protocol node \
  --semantic \
  --semantic-cache cache/primekg-relations.pt \
  --edge-relation-mode concat \
  --epochs 200 \
  --output results/primekg-edge-aware-sage-node.json
```

### Gated edge-aware GraphSAGE

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input drkg.tsv \
  --split-protocol node \
  --semantic \
  --semantic-cache cache/drkg-relations.pt \
  --edge-relation-mode gated \
  --epochs 200 \
  --output results/drkg-edge-aware-gated-node.json
```

### Basis-mixture edge-aware GraphSAGE

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input drkg.tsv \
  --split-protocol node \
  --semantic \
  --semantic-cache cache/drkg-relations.pt \
  --edge-relation-mode basis_mixture \
  --num-relation-bases 4 \
  --epochs 200 \
  --output results/drkg-edge-aware-basis-node.json
```

### DRKG smoke test

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input drkg.tsv \
  --max-edges 1000 \
  --epochs 1 \
  --output /tmp/drkg-smoke.json
```

### DRKG relation cache precompute

```bash
.venv/bin/python -m kgml_new.scripts.generate_relation_embeddings \
  --input drkg.tsv \
  --output cache/drkg-relations.pt \
  --semantic
```

### TxGNN

```bash
.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --input kg.csv \
  --relation indication \
  --epochs 20 \
  --output results/txgnn.json
```

### Node classification on a graph pickle

```bash
.venv/bin/python -m kgml_new.scripts.run_node_classification \
  --graph /path/to/graph.pkl \
  --method edge_sage \
  --semantic \
  --semantic-cache cache/relation_emb.pkl \
  --edge-relation-mode basis_mixture \
  --num-relation-bases 4 \
  --label-attr label \
  --out results/node-classification.json
```

## CLI Arguments Agents Commonly Need

For `run_gpu_method`, the most important flags are:

- `--method`
- `--input`
- `--input-format {auto,csv,pickle}`
- `--max-edges`
- `--source-col`
- `--target-col`
- `--relation-col`
- `--source-type-col`
- `--target-type-col`
- `--split-protocol {node,edge}`
- `--negative-sampling-mode {global,type_matched}`
- `--negatives-per-pos`
- `--decoder {dot,mlp}`
- `--shuffle-relations`
- `--semantic` or `--no-semantic`
- `--semantic-cache`
- `--strict-semantic`
- `--edge-relation-mode {concat,gated,basis_mixture}`
- `--num-relation-bases`
- `--epochs`
- `--seed`
- `--in-dim`
- `--output`

For `run_txgnn`, the most important flags are:

- `--input`
- `--input-format`
- `--max-edges`
- `--relation`
- `--source-node-type`
- `--target-node-type`
- `--epochs`
- `--batch-size`
- `--neg-samples`
- `--seed`
- `--in-dim`
- `--output`

## Output Structure

`run_gpu_method` writes JSON with fields like:

- method metadata
- device metadata
- input metadata
- split protocol metadata
- semantic/cache metadata
- edge relation mechanism metadata
- relation diversity summary
- training history
- validation metrics
- test metrics

Important fields to read first:

- `split_protocol`
- `negative_sampling_mode`
- `semantic`
- `semantic_cache`
- `relation_init`
- `edge_relation_mode`
- `num_relation_bases`
- `max_edges`
- `relation_diversity_summary`
- `test_metrics`

`run_txgnn` writes similar summary JSON but with typed-edge metadata and the chosen target relation.

## Reproducibility Guidance

When running experiments, always record:

- exact command
- dataset path
- whether input was a prefix slice, random sample, or stratified sample
- split protocol
- negative sampling mode
- semantic or random relation initialization
- semantic cache path
- epoch count
- seed
- output path

If `--max-edges` is used on DRKG, also record:

- whether the slice was a prefix
- if not a prefix, the exact row range or sampling procedure

## Practical Agent Advice

Before launching a semantic-ablation experiment, verify:

1. how many distinct relations are present in the actual slice being used
2. whether the slice is a prefix or a sampled subset
3. whether the cache already exists and matches the intended slice
4. whether CUDA is available
5. whether the comparison is really about semantic relation content, or accidentally about a single predicate

Before interpreting results, verify:

1. whether `relation_diversity_summary` is meaningful
2. whether the model used the learned relation projection
3. whether the split protocol is `node` or `edge`
4. whether the negative sampling mode changed automatically

## Known Limitations

- DRKG prefix slices are misleading for semantic studies because of row ordering
- semantic-vs-random comparison is weaker when the slice has only one relation type
- current semantic setup freezes the raw relation table but still learns a projection layer
- `node2vec` is not a fair main baseline under node-disjoint evaluation

## If You Need To Extend The Repo

Likely extension points:

- add a PrimeKG glossary analogous to `relation_glossary.tsv`
- add a sampling utility for relation-balanced DRKG subsets
- add an explicit flag to freeze the relation projection layer
- add experiment scripts that save the exact row range or sample indices used
- add richer result manifests grouping command, sample, cache, and metrics together
