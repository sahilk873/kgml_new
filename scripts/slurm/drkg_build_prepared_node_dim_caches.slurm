#!/bin/bash
# CPU: build prepared link-prediction caches for node-split semantic-vs-random sweep (in_dim 128 and 256).
# No GPU — run this before the GPU array, or use submit_drkg_node_semantic_random_dim_sweep.sh (chains afterok).
#
# Outputs (default SPLIT=node, SEED=42):
#   cache/drkg-prepared_node_s${SEED}_d128.pkl
#   cache/drkg-prepared_node_s${SEED}_d256.pkl
#
# Same knobs as drkg_build_prepared_cache.slurm / export_prepared_link_prediction.
# Override dims: NODE_PREP_DIMS="128 256" (space-separated)
#
#SBATCH --job-name=drkg-prep-node-dim
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/drkg-prep-node-dim-%j.out
#SBATCH --error=logs/slurm/drkg-prep-node-dim-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p logs/slurm cache

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in $REPO" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

SPLIT="${SPLIT:-node}"
SEED="${SEED:-42}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"

if [[ "${INPUT}" != /* ]]; then
  INPUT="${REPO}/${INPUT}"
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input KG not found: ${INPUT}" >&2
  exit 1
fi

SHUFFLE_EXTRA=()
if [[ "${SHUFFLE_RELATIONS:-0}" == "1" ]]; then
  SHUFFLE_EXTRA=(--shuffle-relations)
fi

echo "=== drkg_build_prepared_node_dim_caches host=$(hostname) ==="
echo "SPLIT=${SPLIT} SEED=${SEED} INPUT=${INPUT}"

for DIM in ${NODE_PREP_DIMS:-128 256}; do
  PREPARED_CACHE="${REPO}/cache/drkg-prepared_${SPLIT}_s${SEED}_d${DIM}.pkl"
  echo "--- in_dim=${DIM} -> ${PREPARED_CACHE} ---"

  if [[ -f "${PREPARED_CACHE}" ]] && [[ "${FORCE_REBUILD_PREPARED_CACHE:-0}" != "1" ]]; then
    echo "Cache already exists; skipping (set FORCE_REBUILD_PREPARED_CACHE=1 to rebuild)."
    continue
  fi

  EXPORT_CMD=(
    python -m kgml_new.scripts.export_prepared_link_prediction
    --input "${INPUT}"
    --input-format auto
    --split-protocol "${SPLIT}"
    --seed "${SEED}"
    --in-dim "${DIM}"
    --negatives-per-pos "${NEGATIVES_PER_POS}"
    --decoder "${DECODER}"
    --val-ratio "${VAL_RATIO}"
    --test-ratio "${TEST_RATIO}"
    --output "${PREPARED_CACHE}"
  )
  if [[ -n "${MAX_EDGES:-}" ]]; then
    EXPORT_CMD+=(--max-edges "${MAX_EDGES}")
  fi
  if [[ -n "${NEGATIVE_SAMPLING_MODE:-}" ]]; then
    EXPORT_CMD+=(--negative-sampling-mode "${NEGATIVE_SAMPLING_MODE}")
  fi
  EXPORT_CMD+=("${SHUFFLE_EXTRA[@]}")

  set -x
  "${EXPORT_CMD[@]}"
  set +x
  echo "Wrote ${PREPARED_CACHE}"
done

echo "=== drkg_build_prepared_node_dim_caches done ==="
