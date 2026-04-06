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
| `submit_drkg_full_with_cache.sh` | Submits CPU build, then GPU array with `afterok` dependency. |
| `drkg_gpu_common.sh` | **Sourced only:** shared env/path/prepared-cache logic for single-method GPU jobs. |
| `drkg_baseline_gcn.slurm` | Single GPU: `baseline_gcn` + prepared cache. |
| `drkg_node2vec.slurm` | Single GPU: `node2vec` + prepared cache (`NODE2VEC_EPOCHS` optional). |
| `drkg_edge_aware_gated.slurm` | Single GPU: `edge_aware_sage`, `--edge-relation-mode gated` + SapBERT cache. |
| `drkg_edge_aware_basis_mixture.slurm` | Single GPU: `basis_mixture` + SapBERT; `NUM_RELATION_BASES`, `SEMANTIC_ALIGNMENT_LAMBDA`. |
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

## Recommended: cache on CPU, then sweep on GPU

```bash
# From repo root; set DRKG_INPUT / SPLIT / SEED to match your experiment
export DRKG_INPUT=cache/drkg.graph.pkl   # or drkg.tsv
./scripts/slurm/submit_drkg_full_with_cache.sh
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

- **Prepared cache missing / metadata mismatch:** Rebuild with `drkg_build_prepared_cache.slurm` using the **same** knobs as training, or align CLI with the stored meta (see error text from `run_gpu_method`).
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
