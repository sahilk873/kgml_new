# Experiments Guide

This repository supports two main experiment families:

1. Generic link prediction on a homogeneous `networkx.Graph` view of the KG.
2. TxGNN-style typed link prediction on a heterogeneous graph.

The default benchmark for link prediction is the node-disjoint split. That is the main setting to use when comparing methods intended to generalize to unseen entities.

For cache formats, embedding APIs, DRKG caveats, and **checklists for code changes**, see [`AGENTS.md`](AGENTS.md).

## End-to-end experimental procedure

Use this sequence for a reproducible link-prediction run:

1. **Environment:** `pip install -e ".[dev]"` and, if using OpenAI relation embeddings, `pip install -e ".[semantic]"` plus `OPENAI_API_KEY` (and optionally `.env` via `python-dotenv`).
2. **Choose input:** PrimeKG CSV (`data/kg.csv`), DRKG TSV (`drkg.tsv`), another table (set `--source-col` / `--target-col` / `--relation-col`), or a pickle (`--input-format pickle`). Remember DRKG prefix slices are often single-relation unless you shuffle or sample (see below).
3. **Optional relation cache (edge-aware / semantic):** Precompute once with `generate_relation_embeddings` (see [Relation embedding cache generator](#relation-embedding-cache-generator)). Match `--max-edges` and relation vocabulary to the graph slice you will train on.
4. **Run one method:** `python -m kgml_new.scripts.run_gpu_method --method ... --output results/....json` with explicit `--split-protocol`, `--seed`, `--epochs`, and any `--semantic` / `--semantic-cache` / `--embedding-model` settings.
5. **Record metadata:** Save the exact shell command, dataset path, whether `--max-edges` was a prefix slice, cache path, and JSON output path (the result file already contains many of these fields—verify them).
6. **Sanity-check:** Run the [verification](#verification-smoke-tests-and-tests) commands after environment or code changes.

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

Optional semantic relation embeddings for edge-aware runners:

```bash
pip install -e ".[semantic]"
export OPENAI_API_KEY=...
```

Semantic prompts use `relation_glossary.tsv` for DRKG-style sourced predicates and fall back to the raw relation string otherwise. PrimeKG-style labels stay raw unless you add a separate PrimeKG glossary.
Semantic caches keep the full OpenAI embedding width; the edge-aware model learns a projection into the configured message-space `edge_dim`.
Legacy semantic caches saved before the full-width change are automatically regenerated when reused.

### GPU and CUDA alignment (when the venv “does not match” the GPU)

Symptoms include: `torch.cuda.is_available()` is false on a GPU node, CUDA kernel errors, `CUDA driver version is insufficient`, failed loads of `torch_scatter`/`torch_sparse`/`pyg_lib` `.so` files, or training that only ever uses CPU.

**Root causes (typical on clusters):**

1. **Wrong node** — venv is fine but the job/login session has no GPU (`nvidia-smi` missing or “no devices”). Run training on a GPU partition and request a GPU.
2. **CPU-only PyTorch** — `pip install torch` resolved to a CPU wheel. Install the CUDA build your site documents (PyTorch “Start locally” picker: Linux + Pip + your CUDA tag).
3. **CUDA tag vs driver** — Wheels are labeled `+cu118`, `+cu121`, `+cu124`, `+cu130`, etc. The **NVIDIA driver** on the node must be **new enough** for that PyTorch build (see [PyTorch compatibility](https://pytorch.org/get-started/locally/)). If the driver is too old, install an older `cuXXX` torch, or use a newer driver / different queue.
4. **PyG extensions out of sync** — After **any** torch reinstall, reinstall `torch_scatter` and `torch_sparse` from the PyG index that matches `torch.__version__` (same `+cu…` suffix). `pyg_lib` is optional and may fail on older glibc (RHEL8); mismatched extensions cause import errors or subtle runtime failures.
5. **Filesystem / OS mismatch** — A `.venv` copied from another OS or very old `glibc` can load the wrong binaries. Prefer creating the venv on the **same OS family** as the GPU nodes, or use a container/conda stack from the cluster docs.

**What to run (on a GPU node, venv active):**

```bash
source .venv/bin/activate
./scripts/diagnose_gpu_env.sh
```

**Slurm (full DRKG sweep, one GPU per array task):** from repo root, adjust `#SBATCH` partition/account/mem in `scripts/slurm/drkg_full_experiments_array.slurm`, then:

```bash
sbatch scripts/slurm/drkg_full_experiments_array.slurm
# or: ./scripts/slurm/submit_drkg_full_array.sh
./scripts/slurm/monitor_slurm_jobs.sh JOBID
```

Runs baseline GraphSAGE (mean + max pool) and edge-aware FiLM (mean + max) with `--log-file logs/slurm/run-drkg-${JOB}_${TASK}.log` and `--require-cuda`. Set `REL_CACHE` (default `cache/drkg-relations-sapbert.pt`) before `sbatch` if needed.

For **node split**, **OOD difficulty metrics**, optional **relation holdout**, and **SapBERT vs random** FiLM controls, use `scripts/slurm/drkg_ood_interpretability_array.slurm` (or `./scripts/slurm/submit_drkg_ood_interpretability.sh`). Details: `scripts/slurm/README.md` (section “OOD interpretability bundle”).

If Slurm exits immediately with `torch.cuda.is_available() is False` while `nvidia-smi` works, your pip PyTorch build is likely **too new for the node driver** (e.g. `+cu130` on a CUDA 12.8 driver). Reinstall once from the venv:

```bash
source .venv/bin/activate
bash scripts/reinstall_torch_cu124.sh
```

The batch script now checks CUDA before loading DRKG so jobs fail fast with that hint.

If CUDA is available but PyG extensions fail, reinstall them after torch is correct:

```bash
pip uninstall -y pyg_lib torch_scatter torch_sparse 2>/dev/null || true
./scripts/install_pyg_extensions.sh
```

Edge-aware methods support three relation-control modes:

- `concat`: baseline path, projected relation embedding is concatenated into the message
- `gated`: projected relation embedding produces a sigmoid gate over the learned neighbor message
- `basis_mixture`: projected relation embedding produces mixture weights over `K` shared basis message transforms

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

### HGT runner (standalone)

Path: `python -m kgml_new.scripts.run_hgt` (or `kgml-new-hgt`)

Same input requirements as the TxGNN runner (typed CSV/TSV or pickled `networkx.Graph`). Uses [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) layers (`torch_geometric.nn.HGTConv`) plus DistMult scoring—separate implementation from TxGNN. Useful flags: `--num-heads` (default 4), `--num-layers` (default 2), `--dropout`. Channel dimensions are aligned to be divisible by `num_heads`.

**Neighbor sampling (large graphs):** By default, training and validation/test scoring use PyG `LinkNeighborLoader` mini-batches (`--neighbor-sampling`, on when `torch_sparse` is available—install `torch_scatter` and `torch_sparse` from the PyG wheel index matching your CUDA PyTorch). On older glibc (e.g. RHEL8), skip optional `pyg_lib` if import fails; `torch_sparse` alone is sufficient. Tune `--batch-size`, `--eval-batch-size`, and `--num-neighbors`. Use **`scripts/bootstrap_gpu_venv.sh`** (`RECREATE=1`) for a reproducible `.venv`. Use `--no-neighbor-sampling` only for tiny graphs or when the sampler backend is missing (falls back to full-graph encode and may OOM).

**Splits (link prediction):** `--split-protocol {edge,node}` selects random edge-disjoint vs node-disjoint evaluation (defaults mirror `export_prepared_link_prediction`: `--val-ratio 0.1`, `--test-ratio 0.1`). Typed bipartite relations (e.g. PrimeKG drug–disease) use an internal bipartite node split; same-type relations (typical FB15k-237 entity–entity per predicate) use the homogeneous node split. `--negative-sampling-mode {global,type_matched}` controls val/test negatives (default `global`). Result JSON includes `split_protocol`, `val_ratio`, `test_ratio`, `bipartite_node_split`, and `negative_sampling_mode`.

**FB15k headerless TSV:** pass `--source-col source --relation-col relation --target-col target` (see loader), and `--relation` with the **exact** predicate string in column 2.

Slurm examples:

- Single GPU (default PrimeKG-style `data/kg.csv`): `scripts/slurm/run_hgt.slurm`.
- PrimeKG + FB15k × edge/node array: `scripts/slurm/hgt_primekg_fb15k_split_array.slurm` (submit via `scripts/slurm/submit_hgt_primekg_fb15k.sh`). Set `HGT_RELATION_FB15K237` before submit.

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

### Unseen relation types (relation holdout)

Pass one or more relation names with `--held-out-relations` so those relation types **never appear as training positives**. They are split only into validation and test queries. Seen relation types use the usual node or edge split on the remaining edges. This tests whether the model generalizes to **zero-shot relation types** (especially when comparing semantic relation embeddings to random or shuffled controls).

**Choosing relations:** Pick held-out types with enough val/test edges to stabilize metrics; keep training relations non-trivial. Semantically related seen vs held-out pairs (e.g. agonist / activator / binds vs antagonist / inhibitor / blocker) make a stronger story. To inspect counts, load the graph and tally edge relations (e.g. `collections.Counter` over the relation column).

**Controls (paper table columns):**

| Column | Typical CLI |
|--------|----------------|
| Random | `--no-semantic` or `--embedding-model random` |
| Shuffled | `--shuffle-relations` plus a semantic cache (relation labels permuted; embeddings still “semantic”) |
| Semantic | `--semantic` with `--semantic-cache` and/or OpenAI/SapBERT as usual |

**Commands:**

```bash
# Edge split, hold out two relation names (repeat flag or space-separated list)
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input data/kg.csv \
  --split-protocol edge \
  --held-out-relations antagonist inhibitor \
  --semantic --semantic-cache cache/primekg-relations.pt \
  --epochs 200 \
  --output results/edge_aware_holdout.json
```

```bash
# Node split (default) with the same holdout list
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input data/kg.csv \
  --split-protocol node \
  --held-out-relations antagonist inhibitor \
  --semantic --semantic-cache cache/primekg-relations.pt \
  --output results/edge_aware_holdout_node.json
```

**Prepared cache:** `export_prepared_link_prediction.py` accepts the same `--held-out-relations`; cache metadata records them for `run_gpu_method --prepared-dataset-cache`.

**Reading results:** The JSON field `relation_holdout` contains `val` and `test`, each with `seen_relations` and `unseen_relations` metric dicts (ROC-AUC, AP, hits@k) plus `n_seen_queries` / `n_unseen_queries`. Top-level `held_out_relations` / `held_out_relation_ids` echo the resolved holdout set. If semantic embeddings help **unseen** relations more than **seen**, that supports using relation **meaning** beyond memorized relation IDs.

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
- `edge_aware_link_mlp`

### OOD difficulty analysis (optional)

Add post-hoc stratified metrics (node frequency, novelty, distance to train support, relation frequency, support, combined difficulty) to the run JSON, and optionally export per-edge rows for plotting. Not available for `node2vec` in the current runner.

Minimal example:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input data/kg.csv \
  --split-protocol node \
  --output results/run.json \
  --compute-ood-difficulty \
  --save-edge-predictions
```

Full flag list, graph policy, aggregation across seeds, and `pytest` coverage are documented in [AGENTS.md](AGENTS.md) under **`run_gpu_method` → OOD difficulty analysis**.

### Common commands

PrimeKG, node-disjoint baseline GraphSAGE:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input data/kg.csv \
  --split-protocol node \
  --epochs 200 \
  --output results/baseline_sage-node.json
```

PrimeKG, legacy edge split:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input data/kg.csv \
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
  --input data/kg.csv \
  --split-protocol node \
  --semantic \
  --semantic-cache cache/primekg-edge-aware-semantic.pt \
  --edge-relation-mode concat \
  --epochs 200 \
  --output results/edge_aware_sage-node.json
```

Relation-gated GraphSAGE:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input drkg.tsv \
  --split-protocol node \
  --semantic \
  --semantic-cache cache/drkg-relations.pt \
  --edge-relation-mode gated \
  --epochs 200 \
  --output results/edge_aware_sage-gated-node.json
```

Relation-basis-mixture GraphSAGE:

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
  --output results/edge_aware_sage-basis-node.json
```

Precompute a DRKG glossary-first OpenAI relation cache once and reuse it:

```bash
.venv/bin/python -m kgml_new.scripts.generate_relation_embeddings \
  --input drkg.tsv \
  --output cache/drkg-relations.pt \
  --embedding-model openai
```

Random relation cache (no API calls), same relation set:

```bash
.venv/bin/python -m kgml_new.scripts.generate_relation_embeddings \
  --input drkg.tsv \
  --output cache/drkg-relations-random.pt \
  --embedding-model random \
  --edge-dim 32
```

Node2Vec:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method node2vec \
  --input data/kg.csv \
  --split-protocol edge \
  --epochs 20 \
  --output results/node2vec-edge.json
```

`node2vec` is not part of the main node-disjoint comparison because unseen nodes do not receive inductive embeddings. The script emits `comparable_under_node_split: false` under node split for that reason.

Link MLP scorer:

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method link_mlp \
  --input data/kg.csv \
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
- `--split-protocol {node,node_category,edge}`: main evaluation choice (`node_category` requires `--held-out-node-categories`)
- `--negative-sampling-mode {global,type_matched}`: defaults to `type_matched` under node split, `global` under edge split
- `--negatives-per-pos`: number of negatives per positive in validation/test
- `--decoder {dot,mlp}`: link scoring head
- `--shuffle-relations`: relation-label ablation
- `--held-out-relations REL [REL ...]`: exclude relation types from training positives (zero-shot relations at val/test); see [Unseen relation types](#unseen-relation-types-relation-holdout)
- `--semantic` / `--no-semantic`: for edge-aware methods, semantic is on by default; `--no-semantic` uses random relation embeddings unless you set `--embedding-model` explicitly
- `--embedding-model {openai,gemini,sapbert,e5,random}`: relation embedding backend (defaults: `openai` when `--semantic`, `random` when `--no-semantic`; override e.g. `--embedding-model gemini` or `sapbert` while keeping `--semantic`)
- `--semantic-cache`: optional `.pt` cache for relation embeddings (must match model slice / relation vocabulary when possible)
- `--glossary-path`: override default `relation_glossary.tsv` for DRKG-style prompts
- `--sapbert-model`: HuggingFace id when `--embedding-model sapbert`
- `--strict-semantic`: fail instead of silently falling back if embedding setup fails
- `--edge-relation-mode {concat,gated,basis_mixture,film}`: how projected relation embeddings control message passing (`film` is used by `edge_aware_sage_film_semdec`; the runner forces `film` for that method)
- `--num-relation-bases`: number of shared basis transforms for `basis_mixture`
- `--semantic-alignment-lambda`: optional regularizer for `basis_mixture` when semantic embeddings are enabled (see `run_gpu_method --help`)
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
- defaults to OpenAI semantic relation embeddings in `run_gpu_method` when `--semantic` is on; use `--embedding-model sapbert` or `random` to change the backend
- legacy `run_link_prediction` uses `--semantic` for OpenAI only; prefer `run_gpu_method` for full embedding-model control
- use `--no-semantic` for the random-initialized ablation (or `--embedding-model random`)
- semantic prompts use `relation_glossary.tsv` for DRKG-style sourced predicates and fall back to raw relation strings otherwise
- `concat` is the current baseline
- `gated` uses the relation embedding to gate the learned neighbor message
- `basis_mixture` uses the relation embedding to mix over `K` learned basis message transforms

`node2vec`

- transductive embedding baseline
- valid for edge split
- not a main comparable baseline under node-disjoint split

`link_mlp`

- trains a baseline GraphSAGE encoder first
- then trains an MLP scorer on pair features

`edge_aware_link_mlp`

- trains an edge-aware GraphSAGE encoder first
- then trains an MLP scorer on pair features
- use this when you want to separate "better encoder" from "better decoder" while keeping relation-aware message passing

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
  --edge-relation-mode gated \
  --split-protocol node \
  --epochs 20 \
  --out artifacts/edge_sage.pt
```

Important arguments:

- `--graph`: pickled `networkx.Graph`
- `--model {sage,edge_sage}`
- `--semantic / --no-semantic`
- `--cache`: relation embedding cache
- `--edge-relation-mode {concat,gated,basis_mixture}`
- `--num-relation-bases`
- `--split-protocol {node,edge}`
- `--negative-sampling-mode {global,type_matched}`
- `--negatives-per-pos`
- `--decoder {dot,mlp}`
- `--shuffle-relations`
- `--epochs`
- `--seed`
- `--out`

## Relation Embedding Cache Generator

Entry point:

```bash
python -m kgml_new.scripts.generate_relation_embeddings --help
```

This script creates the cached relation embedding table used by the edge-aware semantic runners.
The same semantic cache can be reused across `concat`, `gated`, and `basis_mixture` runs because the cache stores relation embeddings only, not the message-passing mechanism.

Example for DRKG (OpenAI):

```bash
.venv/bin/python -m kgml_new.scripts.generate_relation_embeddings \
  --input drkg.tsv \
  --output cache/drkg-relations.pt \
  --embedding-model openai
```

Example for PrimeKG (OpenAI):

```bash
.venv/bin/python -m kgml_new.scripts.generate_relation_embeddings \
  --input data/kg.csv \
  --output cache/primekg-relations.pt \
  --embedding-model openai
```

Important arguments:

- `--input`: CSV, TSV, or pickle path
- `--input-format {auto,csv,pickle}`
- `--output`: `.pt` cache file to create
- `--embedding-model {openai,gemini,sapbert,e5,random}`: **required choice**—there is no `--semantic` flag on this script
- `--relation-text-mode {raw,canonical}`: prompt style for embedding text
- `--edge-dim`: width for **random** relation vectors; OpenAI/SapBERT caches store full model width
- `--max-edges`: useful for smoke tests and slice-matched caches
- `--strict-embedding`: fail instead of falling back when embedding fails
- `--glossary-path`: override the default relation glossary path
- `--sapbert-model`: HuggingFace model name when using SapBERT

## TxGNN Experiments

Entry point:

```bash
python -m kgml_new.scripts.run_txgnn --help
```

PrimeKG example:

```bash
.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --input data/kg.csv \
  --relation indication \
  --epochs 20 \
  --output results/txgnn-indication.json
```

When a relation name appears across multiple typed edges, disambiguate it:

```bash
.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --input data/kg.csv \
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
- `--semantic / --no-semantic`: for `edge_sage`
- `--semantic-cache`: relation embedding cache for `edge_sage`
- `--strict-semantic`: fail instead of silently falling back for `edge_sage`
- `--edge-relation-mode {concat,gated,basis_mixture}`: for `edge_sage`
- `--num-relation-bases`: for `basis_mixture`
- `--epochs`
- `--classifier-epochs`
- `--seed`
- `--in-dim`
- `--hidden-dim`
- `--embedding-dim`
- `--edge-dim`
- `--batch-size`
- `--out`

Edge-aware node classification example:

```bash
.venv/bin/python -m kgml_new.scripts.run_node_classification \
  --graph data/graph.pkl \
  --method edge_sage \
  --semantic \
  --semantic-cache cache/relation_emb.pkl \
  --edge-relation-mode basis_mixture \
  --num-relation-bases 4 \
  --label-attr label \
  --epochs 20 \
  --classifier-epochs 20 \
  --out results/node_cls_edge_basis.json
```

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

Edge-aware runs also record:

- `relation_init`
- `edge_relation_mode`
- `num_relation_bases`
- `semantic`
- `semantic_cache`

With `--held-out-relations`, runs also include `held_out_relations`, `held_out_relation_ids`, and `relation_holdout` (per-split `seen_relations` vs `unseen_relations` metrics).

For node-disjoint link prediction, `metrics` is the test-set metrics.

## Recommended Experiment Matrix

For the main link-prediction study, run:

1. `baseline_sage` with `--split-protocol node`
2. `baseline_gcn` with `--split-protocol node`
3. `edge_aware_sage` with `--split-protocol node`
4. `link_mlp` with `--split-protocol node`
5. `edge_aware_link_mlp` with `--split-protocol node`
6. rerun `edge_aware_sage` with `--edge-relation-mode gated`
7. rerun `edge_aware_sage` with `--edge-relation-mode basis_mixture`

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

## Verification (smoke tests and tests)

After changing dependencies or Python code, run:

```bash
pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

Quick link-prediction smoke (CPU-friendly):

```bash
.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --input drkg.tsv \
  --max-edges 1000 \
  --epochs 1 \
  --output /tmp/drkg-smoke.json

.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method edge_aware_sage \
  --input drkg.tsv \
  --max-edges 500 \
  --epochs 1 \
  --no-semantic \
  --output /tmp/drkg-edge-smoke.json
```

**Node2Vec:** PyTorch Geometric’s `Node2Vec` optionally needs `pyg-lib` or `torch-cluster`. If those are missing, the code falls back to an internal lightweight trainer; for very small dense graphs (e.g. a triangle), negatives may include neighbors as a fallback so training does not crash.

## Practical Notes

- Use `--max-edges` first for smoke tests before launching full runs.
- `run_gpu_method` is the easiest entrypoint for raw PrimeKG CSV and raw DRKG TSV.
- `run_gpu_method` is the main end-to-end link prediction runner: it loads the graph, builds the split, trains one chosen method, evaluates it, and writes a JSON summary.
- `run_link_prediction` is better when you already have a pickled graph and want semantic relation embeddings.
- Under node split, a method must support inductive embeddings for unseen nodes to be part of the main comparison.
- DRKG can be extremely large. Start with subsets to validate runtime and memory on your machine.
- `drkg.tsv` is ordered by relation block, so prefix slices can be low-diversity or single-relation. Prefer shuffled, random, or stratified DRKG subsets for semantic-vs-random comparisons.
- If you already have a full DRKG semantic cache such as `cache/drkg-relations.pt`, you usually do not need to rebuild it for shuffled DRKG subsets as long as the subset relation names are covered.
