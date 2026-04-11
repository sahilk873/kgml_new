# DRKG Slurm workflows (prepared cache, sweeps, extra methods)

This folder contains GPU and CPU Slurm scripts for **DRKG** (and compatible KGs) **link prediction** using `python -m kgml_new.scripts.run_gpu_method`. Read this before changing scripts or debugging jobs.

For general experiment conventions (splits, CLI), see [`EXPERIMENTS.md`](../../EXPERIMENTS.md) and [`AGENTS.md`](../../AGENTS.md).

---

## Concepts

### Prepared dataset cache

Heavy steps (CSV/pickle load, graph build, train/val/test split, negative sampling, PyG tensors) can be exported once to a **pickle** consumed by every GPU run via `--prepared-dataset-cache`.

- **Builder:** `python -m kgml_new.scripts.export_prepared_link_prediction` (wrapped by `drkg_build_prepared_cache.slurm`).
- **Consumer:** `run_gpu_method --prepared-dataset-cache /path/to/cache.pkl`.
- **Metadata:** Loading checks that CLI options (split, seed, `in_dim`, decoder, negatives, shuffle, max edges, negative sampling mode, input fingerprint) match what was stored in the cache. Mismatches fail fast.

Default cache path when `PREPARED_CACHE` is unset:

`cache/drkg-prepared_${SPLIT}_s${SEED}.pkl`  
Example: `SPLIT=edge`, `SEED=42` → `cache/drkg-prepared_edge_s42.pkl`.

### Relation embedding cache (SapBERT / OpenAI / random)

**Edge-aware** methods need a **relation table** built from the graph. For DRKG sweeps we usually point `--semantic-cache` at a precomputed file (default in scripts: `cache/drkg-relations-sapbert.pt`). Build once, e.g.:

```bash
python -m kgml_new.scripts.generate_relation_embeddings \
  --input "$DRKG_INPUT" --output cache/drkg-relations-sapbert.pt --embedding-model sapbert
```

### Why CPU cache build + GPU dependency

GPU jobs should **not** sit idle while one array task builds the cache. Pattern:

1. Submit a **CPU** job that only builds (or skips) the prepared pickle.
2. Submit GPU work with `--dependency=afterok:<cpu_job_id>` so GPUs start only after a successful cache.

`submit_drkg_full_with_cache.sh` automates this.

---

## Scripts (quick reference)

| Script | Role |
|--------|------|
| `drkg_build_prepared_cache.slurm` | **CPU:** build or skip prepared cache (`FORCE_REBUILD_PREPARED_CACHE=1` to rebuild). |
| `drkg_full_experiments_array.slurm` | **GPU array 0–3:** baseline SAGE (mean/max), edge-aware FiLM + SapBERT (mean/max). **Requires** prepared cache if `USE_PREPARED_DATASET_CACHE=1` (default). |
| `drkg_ood_interpretability_array.slurm` | **GPU array 0–5:** same baselines + FiLM **semantic vs random** (4–5), default **`SPLIT=node`**, always **`--compute-ood-difficulty`**, optional holdout via `HELD_OUT_RELATIONS`. |
| `submit_drkg_ood_interpretability.sh` | Submits CPU `drkg_build_prepared_cache` (same `SPLIT` / holdout defaults), then OOD array with `afterok`. |
| `submit_drkg_full_with_cache.sh` | Submits CPU build, then GPU array with `afterok` dependency. |
| `drkg_gpu_common.sh` | **Sourced only:** shared env/path/prepared-cache logic for single-method GPU jobs. |
| `drkg_baseline_gcn.slurm` | Single GPU: `baseline_gcn` + prepared cache. |
| `drkg_node2vec.slurm` | Single GPU: `node2vec` + prepared cache (`NODE2VEC_EPOCHS` optional). |
| `drkg_edge_aware_gated.slurm` | Single GPU: `edge_aware_sage`, `--edge-relation-mode gated` + SapBERT cache. |
| `drkg_edge_aware_basis_mixture.slurm` | Single GPU: `basis_mixture` + SapBERT; `NUM_RELATION_BASES`, `SEMANTIC_ALIGNMENT_LAMBDA`. |
| `drkg_context_injection_ablation_array.slurm` | GPU array (8 tasks, one split): context-injection ablations over `concat/gated/basis_mixture/film` x `random/shuffled_semantic`, fixed full-run defaults. |
| `submit_drkg_context_injection_ablation.sh` | One-command submit helper for BOTH splits (`edge` and `node`) for the context-injection ablation array. |
| `drkg_node_semantic_backfill_array.slurm` | GPU array (4 tasks, node split): semantic backfill for missing `film/concat/gated/basis_mixture` runs with `neighbor_aggr=max`. |
| `submit_drkg_node_semantic_backfill.sh` | Submit helper for the node-split semantic backfill array with required cache checks. |
| `drkg_build_prepared_node_dim_caches.slurm` | **CPU:** builds `cache/drkg-prepared_node_s<SEED>_d128.pkl` and `_d256.pkl` for the node semantic-vs-random sweep (no GPU). |
| `drkg_node_semantic_random_dim_sweep_array.slurm` | GPU array (36 tasks, node split default): baseline mean/pool + edge-aware `{concat,gated,basis_mixture,film}` with mean/pool, dims `128/256`, controls `sapbert/random`. **Requires** per-dim prepared caches; does not run export on GPU. |
| `submit_drkg_node_semantic_random_dim_sweep.sh` | Submits CPU dim-cache build then GPU array with `afterok` (or `SKIP_PREPARED_CACHE_BUILD=1` for GPU-only). |
| `submit_drkg_prepared_extra_methods.sh` | Submits the four single-method jobs above (cache must already exist). |
| `monitor_slurm_jobs.sh` | Polls until job ends; prints `sacct`; tails stderr / run logs on failure. |

Legacy one-off templates (`run_baseline_sage_gpu.slurm`, etc.) may still point at old paths; prefer the `drkg_*` scripts for DRKG + prepared cache.

---

## Main sweep: `drkg_full_experiments_array.slurm`

- **Array tasks**
  - `0` — `baseline_sage`, neighbor **mean**
  - `1` — `baseline_sage`, neighbor **max** (pool)
  - `2` — `edge_aware_sage`, **FiLM**, neighbor mean, SapBERT `REL_CACHE`
  - `3` — `edge_aware_sage`, **FiLM**, neighbor max, SapBERT `REL_CACHE`
- **Prepared cache:** If `USE_PREPARED_DATASET_CACHE=1`, the cache file must exist before training; the script does **not** build it and does **not** cross-task wait.
- **Optional diversity sidecar:** If `DRKG_INPUT` is `*.graph.pkl` and `foo.diversity.pkl` sits beside it, `--diversity-buckets-cache` is passed automatically (unless `NO_AUTO_DIVERSITY_CACHE=1`).

Submit from **repository root** (paths resolve to `${SLURM_SUBMIT_DIR}`).

---

## OOD interpretability bundle (`drkg_ood_interpretability_array.slurm`)

Use this when you want **harder entity-inductive runs** (default `SPLIT=node`), **post-hoc OOD difficulty** stratification in every JSON, optional **relation holdout** (unseen relation types at val/test), and **SapBERT vs random** relation embeddings on the same FiLM edge-aware architecture (tasks 2–3 vs 4–5).

### Array tasks

| Task | Config | Output name stem |
|------|--------|-------------------|
| 0 | `baseline_sage` mean | `drkg_ood_bsage_mean_*` |
| 1 | `baseline_sage` max | `drkg_ood_bsage_pool_*` |
| 2 | `edge_aware_sage` FiLM mean + SapBERT | `drkg_ood_easage_film_mean_semantic_*` |
| 3 | `edge_aware_sage` FiLM max + SapBERT | `drkg_ood_easage_film_pool_semantic_*` |
| 4 | `edge_aware_sage` FiLM mean + random rel. emb. | `drkg_ood_easage_film_mean_random_*` |
| 5 | `edge_aware_sage` FiLM max + random rel. emb. | `drkg_ood_easage_film_pool_random_*` |

Each run writes `results/slurm/drkg_ood_<name>_<arrayJobId>_<taskId>.json` with an `ood_difficulty` block. If `HELD_OUT_RELATIONS` is set, outputs also include **`relation_holdout`** metrics for those types.

### Prepared cache and holdout

- With **no** holdout, default cache path matches the main sweep: `cache/drkg-prepared_${SPLIT}_s${SEED}.pkl`.
- With **`HELD_OUT_RELATIONS`** (space-separated relation names) and **unset** `PREPARED_CACHE`, both this GPU script and **`drkg_build_prepared_cache.slurm`** default to:
  - `cache/drkg-prepared_${SPLIT}_s${SEED}_ho_<slug>.pkl`  
  where `<slug>` is derived from the names (alphanumeric + underscore). Build and train **must** use the same names and knobs.

Example CPU build + GPU (or use `submit_drkg_ood_interpretability.sh`):

```bash
export DRKG_INPUT=cache/drkg.graph.pkl
export SPLIT=node
export HELD_OUT_RELATIONS='SomeRelationName'   # optional
BUILD=$(sbatch --parsable --partition=batch scripts/slurm/drkg_build_prepared_cache.slurm)
sbatch --dependency=afterok:${BUILD} scripts/slurm/drkg_ood_interpretability_array.slurm
```

### Relation caches

- **Semantic** tasks: `REL_CACHE` (default `cache/drkg-relations-sapbert.pt`).
- **Random** tasks: `REL_RANDOM_CACHE` (default `cache/drkg-relations-random.pt`). Build once with `generate_relation_embeddings --embedding-model random` if missing.

### OOD CSV / per-edge scores

- **`OOD_SAVE_EDGE_CSV=1`** enables `--save-edge-predictions` (writes under `OOD_OUTPUT_DIR`, default `results/ood`). On full DRKG this is **large**; prefer **`MAX_EDGES`** smoke runs when exporting CSV.

### Aggregating semantic vs random gaps

After runs complete, pair by **seed** and compare **`embedding_model_resolved`** (`sapbert` vs `random`) on bucketed test `roc_auc`:

```bash
python -m kgml_new.scripts.aggregate_ood_results \
  --inputs results/slurm/drkg_ood_*.json \
  --tag drkg_ood_node \
  --baseline-embedding random \
  --target-embedding sapbert \
  --out-dir results/ood
```

Narrow `--inputs` to a single array job’s JSONs if your glob is too broad.

### One-command submit

```bash
./scripts/slurm/submit_drkg_ood_interpretability.sh
```

Overrides: `SPLIT`, `HELD_OUT_RELATIONS`, `DRKG_INPUT`, `BUILD_PARTITION`, `OOD_SAVE_EDGE_CSV`, scheduler vars — same style as other submit helpers (`--export=ALL`).

---

## Recommended: cache on CPU, then sweep on GPU

```bash
# From repo root; set DRKG_INPUT / SPLIT / SEED to match your experiment
export DRKG_INPUT=cache/drkg.graph.pkl   # or drkg.tsv
./scripts/slurm/submit_drkg_full_with_cache.sh
```

**Node-disjoint split** (separate prepared cache `cache/drkg-prepared_node_s<SEED>.pkl`; not the edge cache):

```bash
SPLIT=node ./scripts/slurm/submit_drkg_full_with_cache.sh
```

Manual equivalent:

```bash
BUILD=$(sbatch --parsable --partition=batch scripts/slurm/drkg_build_prepared_cache.slurm)
sbatch --dependency=afterok:${BUILD} scripts/slurm/drkg_full_experiments_array.slurm
```

- **CPU partition:** Many clusters have no partition named `cpu`. On this site, `batch` is used; override when submitting: `BUILD_PARTITION=smp ./scripts/slurm/submit_drkg_full_with_cache.sh`.

---

## Extra single-method GPU runs (GCN, node2vec, gated, basis mixture)

Use when the **prepared cache already exists** (same `SPLIT` / `SEED` / input fingerprint as when the cache was built).

```bash
export DRKG_INPUT=cache/drkg.graph.pkl   # must match cache build input
# PREPARED_CACHE=cache/drkg-prepared_edge_s42.pkl  # optional if default matches
./scripts/slurm/submit_drkg_prepared_extra_methods.sh
```

Or submit one script:

```bash
sbatch scripts/slurm/drkg_baseline_gcn.slurm
```

**Outputs:** `results/slurm/drkg_<method>_<jobid>.json` (exact pattern is in each `.slurm` file).  
**Slurm stdout/stderr:** `logs/slurm/drkg-gcn-<jobid>.out`, etc.  
**Python structured log:** `logs/slurm/run-drkg-<stem>-<jobid>.log` (see each script).

**Edge-aware env knobs**

- `NEIGHBOR_AGGR` — `mean` or `max` (default `mean`).
- `NUM_RELATION_BASES` — basis mixture only (default `4`).
- `SEMANTIC_ALIGNMENT_LAMBDA` — optional alignment regularizer for `basis_mixture` (default `0`).

---

## Context-injection ablation (random vs shuffled semantic; edge + node)

Use this workflow when you want a controlled ablation that keeps major hyperparameters fixed to prior full runs while varying only relation-control mode and context source.

### Coverage

Per split (`edge` or `node`), the array runs 8 tasks:

- `concat_random`, `concat_shuffled_semantic`
- `gated_random`, `gated_shuffled_semantic`
- `basis_mixture_random`, `basis_mixture_shuffled_semantic`
- `film_random`, `film_shuffled_semantic`

Both splits combined: 16 runs (single seed by default).

### Submit both splits in one command

```bash
./scripts/slurm/submit_drkg_context_injection_ablation.sh
```

This helper submits:

1. `SPLIT=edge` array
2. `SPLIT=node` array

using `drkg_context_injection_ablation_array.slurm`.

### Required caches

- Prepared dataset caches:
  - `cache/drkg-prepared_edge_s<SEED>.pkl`
  - `cache/drkg-prepared_node_s<SEED>.pkl`
- Relation caches:
  - `RANDOM_REL_CACHE` (default `cache/drkg-relations-random.pt`)
  - `SHUFFLED_SEM_REL_CACHE` (default `cache/drkg-relations-sapbert-shuffled.pt`)

### Fixed defaults (same as established full runs)

- `EPOCHS=100`
- `SEED=42`
- `IN_DIM=64`
- `DECODER=dot`
- `NEGATIVES_PER_POS=20`
- `USE_PREPARED_DATASET_CACHE=1`

### Optional environment variables

- `DRKG_INPUT`
- `PREPARED_CACHE_EDGE`, `PREPARED_CACHE_NODE`
- `RANDOM_REL_CACHE`, `SHUFFLED_SEM_REL_CACHE`
- `NUM_RELATION_BASES` (basis_mixture only; default `4`)
- `SEMANTIC_ALIGNMENT_LAMBDA` (basis_mixture only; default `0.0`)
- scheduler overrides: `PARTITION`, `ACCOUNT`, `QOS`

### Output files

Each task writes:

`results/slurm/drkg_ctxabl_${SPLIT}_${mode}_${control}_${JOB_TAG}.json`

Where `JOB_TAG` is `${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}`.

### Monitoring

After submit, monitor each array:

```bash
LOG_PREFIX=drkg-ctxabl ./scripts/slurm/monitor_slurm_jobs.sh <EDGE_JOBID> --poll 30
LOG_PREFIX=drkg-ctxabl ./scripts/slurm/monitor_slurm_jobs.sh <NODE_JOBID> --poll 30
```

---

## Node-split semantic backfill (film pool + concat/gated/basis semantic)

Use this workflow to run the missing node-split semantic experiments with existing caches.

### Coverage

Single array (`0-3`) runs:

- `film` semantic, `neighbor_aggr=max` (pool)
- `concat` semantic, `neighbor_aggr=max`
- `gated` semantic, `neighbor_aggr=max`
- `basis_mixture` semantic, `neighbor_aggr=max` (`num_relation_bases=4`, `semantic_alignment_lambda=0.0`)

### Submit

```bash
./scripts/slurm/submit_drkg_node_semantic_backfill.sh
```

or directly:

```bash
sbatch scripts/slurm/drkg_node_semantic_backfill_array.slurm
```

### Required caches / defaults

- Prepared node cache (default): `cache/drkg-prepared_node_s42.pkl`
- Semantic relation cache (default): `cache/drkg-relations-sapbert.pt`
- Fixed defaults aligned with prior full runs:
  - `EPOCHS=100`
  - `SEED=42`
  - `IN_DIM=64`
  - `DECODER=dot`
  - `NEGATIVES_PER_POS=20`
  - `USE_PREPARED_DATASET_CACHE=1`

### Outputs

`results/slurm/drkg_node_semantic_<mode>_pool_<array_jobid>_<taskid>.json`

### Monitoring

```bash
LOG_PREFIX=drkg-node-sem-backfill ./scripts/slurm/monitor_slurm_jobs.sh <ARRAY_JOBID> --poll 30
```

---

## Node split semantic-vs-random divergence sweep (dims 128/256)

Use this workflow to test whether semantic relation context matters versus random relation embeddings under the same architecture and training setup.

### Coverage

Single array (`0-35`) runs:

- Baseline:
  - `baseline_sage x {mean,max} x {128,256}` = 4 tasks
- Edge-aware:
  - `{concat,gated,basis_mixture,film} x {mean,max} x {128,256} x {sapbert,random}` = 32 tasks

Total: **36 tasks**.

### Submit

```bash
./scripts/slurm/submit_drkg_node_semantic_random_dim_sweep.sh
```

Or directly:

```bash
sbatch scripts/slurm/drkg_node_semantic_random_dim_sweep_array.slurm
```

### Defaults and required caches

- `SPLIT=node` (default in script; should match prepared cache metadata)
- Prepared dataset cache (per `--in-dim`, must match training):
  - default `cache/drkg-prepared_node_s<SEED>_d128.pkl` and `cache/drkg-prepared_node_s<SEED>_d256.pkl`
  - **build on CPU before GPU** — GPU tasks do not run `export_prepared_link_prediction` (avoids tying up GPU nodes for CPU-heavy prep):
    - `sbatch --partition=batch scripts/slurm/drkg_build_prepared_node_dim_caches.slurm` (override `BUILD_PARTITION` if your site differs), or
    - `./scripts/slurm/submit_drkg_node_semantic_random_dim_sweep.sh` which submits the CPU build then the GPU array with `--dependency=afterok:<cpu_job>`
  - if caches already exist: `SKIP_PREPARED_CACHE_BUILD=1 ./scripts/slurm/submit_drkg_node_semantic_random_dim_sweep.sh`
  - optional: set `PREPARED_CACHE` to one path for all tasks only if that cache’s `in_dim` matches every task (usually leave unset for this sweep)
  - `FORCE_REBUILD_PREPARED_CACHE=1` on the CPU build script forces rebuild (same semantics as `drkg_build_prepared_cache.slurm`)
- Relation caches (for edge-aware tasks):
  - `SAPBERT_REL_CACHE` default `cache/drkg-relations-sapbert.pt`
  - `RANDOM_REL_CACHE` default `cache/drkg-relations-random.pt`

### Important env variables

- `DRKG_INPUT`, `PREPARED_CACHE`, `SEED`, `EPOCHS`, `DECODER`, `NEGATIVES_PER_POS`
- `NODE_PREP_DIMS` (CPU builder only; default `128 256`)
- `SKIP_PREPARED_CACHE_BUILD` (submit helper only; `1` = GPU sweep only)
- `BUILD_PARTITION` (CPU builder / chained submit; default `batch`)
- `SAPBERT_REL_CACHE`, `RANDOM_REL_CACHE`
- `NUM_RELATION_BASES`, `SEMANTIC_ALIGNMENT_LAMBDA` (basis_mixture tasks)
- scheduler overrides: `PARTITION`, `ACCOUNT`, `QOS`

### Output files

Each task writes:

`results/slurm/drkg_node_semrand_${NAME}_${JOB_TAG}.json`

Where `NAME` encodes method/mode, aggregator, dimension, and control.

### Monitoring

```bash
LOG_PREFIX=drkg-node-semrand ./scripts/slurm/monitor_slurm_jobs.sh <JOBID> --poll 30
```

---

## Environment variables (shared across DRKG Slurm scripts)

Set via `export` before `sbatch`, or `sbatch --export=ALL,VAR=...`.

| Variable | Typical meaning |
|----------|-----------------|
| `DRKG_INPUT` | Graph file (`drkg.tsv`, `*.graph.pkl`, …). |
| `PREPARED_CACHE` | Path to prepared pickle; default derived from `SPLIT`, `SEED`. |
| `USE_PREPARED_DATASET_CACHE` | `1` (default) use cache; `0` full prep on GPU (slow). |
| `SPLIT` | `node` or `edge` (must match cache). |
| `SEED` | RNG seed (must match cache). |
| `EPOCHS` | Training epochs. |
| `IN_DIM`, `DECODER`, `NEGATIVES_PER_POS` | Must match cache metadata. |
| `MAX_EDGES` | Truncate graph (must match cache if set at build time). |
| `NEGATIVE_SAMPLING_MODE` | Optional; must match how cache was built. |
| `SHUFFLE_RELATIONS` | `1` if relations were shuffled at export. |
| `REL_CACHE` | SapBERT (or other) relation embedding file for edge-aware runs. |
| `REL_RANDOM_CACHE` | Random relation embedding `.pt` for `drkg_ood_interpretability_array` tasks 4–5 (default `cache/drkg-relations-random.pt`). |
| `HELD_OUT_RELATIONS` | Space-separated relation names for zero-shot types (must match prepared cache; default cache path gains `_ho_<slug>.pkl` when unset). |
| `OOD_OUTPUT_DIR` | Directory for OOD CSV / run naming (default `results/ood` under repo). |
| `OOD_SAVE_EDGE_CSV` | `1` → `--save-edge-predictions` (disk-heavy on full DRKG). |
| `DIVERSITY_BUCKETS_CACHE` / `NO_AUTO_DIVERSITY_CACHE` | Diversity sidecar (see sweep script comments). |
| `FORCE_REBUILD_PREPARED_CACHE` | `1` forces rebuild in CPU cache script. |
| `BUILD_PARTITION` | CPU queue for `submit_drkg_full_with_cache.sh` (e.g. `batch`, `smp`). |
| `NODE2VEC_EPOCHS` | Overrides `EPOCHS` for `drkg_node2vec.slurm` only. |

---

## Monitoring and troubleshooting

**Queue**

```bash
squeue -u "$USER"
```

**Watch until completion (array / default log names `drkg-full-*`)**

```bash
./scripts/slurm/monitor_slurm_jobs.sh <JOBID>
./scripts/slurm/monitor_slurm_jobs.sh <JOBID> --poll 30
```

**Single-method jobs** use different log stems. Set `LOG_PREFIX` to the `#SBATCH --output` basename without `-<jobid>.out`:

```bash
LOG_PREFIX=drkg-gcn ./scripts/slurm/monitor_slurm_jobs.sh 3237343
# drkg-n2v, drkg-ea-gated, drkg-ea-basis — same pattern
```

**Early failure checks**

```bash
tail -f logs/slurm/drkg-gcn-<jobid>.err
tail -n 80 logs/slurm/run-drkg-gcn-<jobid>.log
```

Common issues:

- **Prepared cache missing / metadata mismatch:** Rebuild with `drkg_build_prepared_cache.slurm` using the **same** knobs as training, or align CLI with the stored meta (see error text from `run_gpu_method`). Input files are matched by **resolved path and byte size**; a changed `mtime` alone (touch, rsync) does not invalidate the cache.
- **`torch.cuda.is_available()` false:** Driver / PyTorch CUDA build mismatch; see `scripts/diagnose_gpu_env.sh` and repo notes on reinstalling torch for the cluster.
- **Invalid Slurm partition:** Edit `#SBATCH --partition` or pass `--partition=...` to `sbatch` for your site.

---

## File layout

```
logs/slurm/          # Slurm .out / .err; run-drkg-*.log for Python
results/slurm/       # JSON metrics from run_gpu_method
cache/               # Prepared pickles, relation caches, optional graph pickles
```

This README is the intended entry point for **agents** and humans running DRKG Slurm experiments in this repository.
