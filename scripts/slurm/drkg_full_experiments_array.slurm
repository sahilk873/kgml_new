#!/bin/bash
# Full DRKG link-prediction sweep on one GPU per array task (Slurm picks free GPUs).
#
# Uses batched GraphSAGE (--require-cuda). Prepared-dataset cache (PyG + split + negatives)
# must already exist — this script does NOT build it and does not hold GPUs while waiting.
# Build on CPU first, then submit this job with a dependency, e.g.:
#   BUILD=$(sbatch --parsable scripts/slurm/drkg_build_prepared_cache.slurm)
#   sbatch --dependency=afterok:${BUILD} scripts/slurm/drkg_full_experiments_array.slurm
# Or: ./scripts/slurm/submit_drkg_full_with_cache.sh
# If the cache file is missing, every array task exits immediately with an error.
# Optional relation-diversity sidecar (--diversity-buckets-cache) when present next to *.graph.pkl.
#
# Array tasks:
#   0 — baseline GraphSAGE, neighbor mean
#   1 — baseline GraphSAGE, neighbor max (pool)
#   2 — edge-aware GraphSAGE, FiLM + neighbor mean + SapBERT relation cache
#   3 — edge-aware GraphSAGE, FiLM + neighbor max (pool) + SapBERT relation cache
#
# Submit from repository root:
#   sbatch scripts/slurm/drkg_full_experiments_array.slurm
#
# Override via environment (export before sbatch, or use --export in sbatch):
#   EPOCHS, SPLIT (node|edge), DRKG_INPUT, REL_CACHE, MAX_EDGES, SEED, IN_DIM,
#   PREPARED_CACHE (default cache/drkg-prepared_${SPLIT}_s${SEED}.pkl; must exist if USE_PREPARED_DATASET_CACHE=1),
#   USE_PREPARED_DATASET_CACHE (0 to disable prepared cache and use live CSV/pickle prep),
#   DECODER, NEGATIVES_PER_POS, VAL_RATIO, TEST_RATIO, SHUFFLE_RELATIONS,
#   DIVERSITY_BUCKETS_CACHE, NO_AUTO_DIVERSITY_CACHE, NEGATIVE_SAMPLING_MODE,
#   PARTITION, QOS, ACCOUNT, MEM_GB, TIME_LIMIT
#
# Default input is ${REPO}/drkg.tsv (full DRKG at repo root). Override if the file lives elsewhere:
#   sbatch --export=ALL,DRKG_INPUT=/path/to/drkg.tsv scripts/slurm/drkg_full_experiments_array.slurm
#
# Optional subset (first N rows of the table; see README for DRKG ordering caveats):
#   sbatch --export=ALL,MAX_EDGES=25000,DRKG_INPUT=data/drkg-shuffled-seed42.tsv ...
#
# Recommended: skip slow CSV→NetworkX on every array task — export pickle + diversity sidecar once,
# then point DRKG_INPUT at the pickle. If INPUT is .../foo.graph.pkl and .../foo.diversity.pkl exists,
# this script passes --diversity-buckets-cache automatically (unless NO_AUTO_DIVERSITY_CACHE=1).
#   python -m kgml_new.scripts.export_graph_pickle \
#     --input drkg.tsv --output cache/drkg.graph.pkl \
#     --write-diversity-buckets cache/drkg.diversity.pkl
#   sbatch --export=ALL,DRKG_INPUT=cache/drkg.graph.pkl scripts/slurm/drkg_full_experiments_array.slurm
#
# Force a specific sidecar path (must exist):
#   sbatch --export=ALL,DRKG_INPUT=cache/drkg.graph.pkl,DIVERSITY_BUCKETS_CACHE=cache/drkg.diversity.pkl ...
#
#SBATCH --job-name=drkg-full
#SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Full DRKG (~6M edges) is memory-heavy during CSV load + graph build; raise if OOM-killed.
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/slurm/drkg-full-%A_%a.out
#SBATCH --error=logs/slurm/drkg-full-%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p logs/slurm results/slurm cache

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in $REPO — create venv and pip install -e . first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

EPOCHS="${EPOCHS:-100}"
SPLIT="${SPLIT:-edge}"
SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-64}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
USE_PREPARED_DATASET_CACHE="${USE_PREPARED_DATASET_CACHE:-1}"
# If unset, omit --negative-sampling-mode so Python picks defaults (node→type_matched, edge→global).
# For tiny smoke graphs only, export NEGATIVE_SAMPLING_MODE=global.
INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"
REL_CACHE="${REL_CACHE:-cache/drkg-relations-sapbert.pt}"

# Always resolve relative data paths against the repo (not the directory you ran sbatch from).
if [[ "${INPUT}" != /* ]]; then
  INPUT="${REPO}/${INPUT}"
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input KG file not found: ${INPUT}" >&2
  echo "  Copy or symlink drkg.tsv into the repo, extract your archive, or:" >&2
  echo "  export DRKG_INPUT=/path/to/drkg.tsv && sbatch ..." >&2
  exit 1
fi

if [[ "${REL_CACHE}" != /* ]]; then
  REL_CACHE="${REPO}/${REL_CACHE}"
fi

# Optional: --diversity-buckets-cache (same node order as graph; build with export_graph_pickle
# --write-diversity-buckets alongside the graph pickle used as DRKG_INPUT).
DIV_EXTRA=()
DIV_CAND=""
if [[ -n "${DIVERSITY_BUCKETS_CACHE:-}" ]]; then
  DIV_CAND="${DIVERSITY_BUCKETS_CACHE}"
elif [[ -z "${NO_AUTO_DIVERSITY_CACHE:-}" ]] && [[ "${INPUT}" == *.graph.pkl ]]; then
  DIV_CAND="${INPUT%.graph.pkl}.diversity.pkl"
fi
if [[ -n "${DIV_CAND}" ]]; then
  if [[ "${DIV_CAND}" != /* ]]; then
    DIV_CAND="${REPO}/${DIV_CAND}"
  fi
  if [[ -f "${DIV_CAND}" ]]; then
    DIV_EXTRA=(--diversity-buckets-cache "${DIV_CAND}")
    echo "Using diversity buckets cache: ${DIV_CAND}"
  elif [[ -n "${DIVERSITY_BUCKETS_CACHE:-}" ]]; then
    echo "ERROR: DIVERSITY_BUCKETS_CACHE not found: ${DIV_CAND}" >&2
    exit 1
  fi
fi

MAX_EXTRA=()
if [[ -n "${MAX_EDGES:-}" ]]; then
  MAX_EXTRA=(--max-edges "${MAX_EDGES}")
fi

PREPARED_EXTRA=()
SHUFFLE_EXTRA=()
if [[ "${SHUFFLE_RELATIONS:-0}" == "1" ]]; then
  SHUFFLE_EXTRA=(--shuffle-relations)
fi
if [[ "${USE_PREPARED_DATASET_CACHE}" == "1" ]]; then
  if [[ -n "${PREPARED_CACHE:-}" ]]; then
    if [[ "${PREPARED_CACHE}" != /* ]]; then
      PREPARED_CACHE="${REPO}/${PREPARED_CACHE}"
    fi
  else
    PREPARED_CACHE="${REPO}/cache/drkg-prepared_${SPLIT}_s${SEED}.pkl"
  fi
  if [[ ! -f "${PREPARED_CACHE}" ]]; then
    echo "ERROR: Prepared dataset cache not found: ${PREPARED_CACHE}" >&2
    echo "  Submit the CPU build job first, then this array with a dependency, e.g.:" >&2
    echo "    BUILD=\$(sbatch --parsable scripts/slurm/drkg_build_prepared_cache.slurm)" >&2
    echo "    sbatch --dependency=afterok:\${BUILD} scripts/slurm/drkg_full_experiments_array.slurm" >&2
    echo "  Or: ./scripts/slurm/submit_drkg_full_with_cache.sh" >&2
    echo "  To skip prepared cache and prep on GPU (slow), set USE_PREPARED_DATASET_CACHE=0." >&2
    exit 1
  fi
  PREPARED_EXTRA=(--prepared-dataset-cache "${PREPARED_CACHE}")
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
LOG="${REPO}/logs/slurm/run-drkg-${JOB_TAG}.log"

echo "=== drkg_full_experiments task=${SLURM_ARRAY_TASK_ID:-?} host=$(hostname) ==="
bash scripts/diagnose_gpu_env.sh || true

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False on this GPU node." >&2
  echo "  Common fix: PyTorch +cu130 needs a newer driver than many CUDA 12.8 clusters." >&2
  echo "  On a login node with this venv: bash scripts/reinstall_torch_cu124.sh" >&2
  echo "  Then: sbatch scripts/slurm/drkg_full_experiments_array.slurm" >&2
  exit 1
fi

echo "Using prepared cache: ${PREPARED_CACHE:-(disabled)}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    NAME=bsage_mean
    EXTRA=(--method baseline_sage --neighbor-aggr mean)
    ;;
  1)
    NAME=bsage_pool
    EXTRA=(--method baseline_sage --neighbor-aggr max)
    ;;
  2)
    NAME=easage_film_mean
    EXTRA=(
      --method edge_aware_sage
      --edge-relation-mode film
      --neighbor-aggr mean
      --semantic-cache "${REL_CACHE}"
      --embedding-model sapbert
    )
    ;;
  3)
    NAME=easage_film_pool
    EXTRA=(
      --method edge_aware_sage
      --edge-relation-mode film
      --neighbor-aggr max
      --semantic-cache "${REL_CACHE}"
      --embedding-model sapbert
    )
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" >&2
    exit 1
    ;;
esac

OUT="${REPO}/results/slurm/drkg_full_${NAME}_${JOB_TAG}.json"

NEG_EXTRA=()
if [[ -n "${NEGATIVE_SAMPLING_MODE:-}" ]]; then
  NEG_EXTRA=(--negative-sampling-mode "${NEGATIVE_SAMPLING_MODE}")
fi

if [[ ! -f "${REL_CACHE}" ]] && [[ "${NAME}" == easage_* ]]; then
  echo "WARNING: REL_CACHE not found at ${REL_CACHE}" >&2
  echo "  Build SapBERT cache once, e.g.:" >&2
  echo "    python -m kgml_new.scripts.generate_relation_embeddings --input \"\${INPUT}\" --output \"\${REL_CACHE}\" --embedding-model sapbert" >&2
  echo "  Or set REL_CACHE to cache/drkg-relations-random.pt and add --no-semantic --embedding-model random to EXTRA in this script." >&2
fi

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --input "${INPUT}" \
  --input-format auto \
  --split-protocol "${SPLIT}" \
  --seed "${SEED}" \
  --in-dim "${IN_DIM}" \
  --decoder "${DECODER}" \
  --negatives-per-pos "${NEGATIVES_PER_POS}" \
  "${SHUFFLE_EXTRA[@]}" \
  "${NEG_EXTRA[@]}" \
  "${MAX_EXTRA[@]}" \
  "${DIV_EXTRA[@]}" \
  "${PREPARED_EXTRA[@]}" \
  --epochs "${EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}" \
  "${EXTRA[@]}"
set +x

echo "Wrote ${OUT}"
