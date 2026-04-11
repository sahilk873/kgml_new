#!/bin/bash
# Build the prepared link-prediction cache (PyG + split + negatives) on CPU — no GPU.
# Submit this job first, then submit the GPU array with Slurm dependency, e.g.:
#   BUILD=$(sbatch --parsable scripts/slurm/drkg_build_prepared_cache.slurm)
#   sbatch --dependency=afterok:${BUILD} scripts/slurm/drkg_full_experiments_array.slurm
# Or: ./scripts/slurm/submit_drkg_full_with_cache.sh
#
# Use the same environment variables as drkg_full_experiments_array.slurm for INPUT,
# SPLIT, SEED, PREPARED_CACHE, etc. (export before sbatch or --export=ALL).
# Optional HELD_OUT_RELATIONS (space-separated names): passed to export and, if PREPARED_CACHE
# is unset, default output becomes cache/drkg-prepared_${SPLIT}_s${SEED}_ho_<slug>.pkl
# (same convention as drkg_ood_interpretability_array.slurm).
#
#SBATCH --job-name=drkg-prep-cache
# On hpc6 the general CPU queue is "batch" (not "cpu"). Override when submitting if needed:
#   sbatch --partition=smp scripts/slurm/drkg_build_prepared_cache.slurm
#   BUILD_PARTITION=smp ./scripts/slurm/submit_drkg_full_with_cache.sh
#SBATCH --partition=batch
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/drkg-prep-cache-%j.out
#SBATCH --error=logs/slurm/drkg-prep-cache-%j.err

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

SPLIT="${SPLIT:-edge}"
SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-64}"
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

if [[ -n "${PREPARED_CACHE:-}" ]]; then
  if [[ "${PREPARED_CACHE}" != /* ]]; then
    PREPARED_CACHE="${REPO}/${PREPARED_CACHE}"
  fi
else
  HO_PART=""
  if [[ -n "${HELD_OUT_RELATIONS:-}" ]]; then
    HO_SLUG=$(printf '%s' "${HELD_OUT_RELATIONS}" | tr -c 'a-zA-Z0-9_' '_' | tr -s '_' | sed 's/^_\|_$//g')
    if [[ -n "${HO_SLUG}" ]]; then
      HO_PART="_ho_${HO_SLUG}"
    fi
  fi
  PREPARED_CACHE="${REPO}/cache/drkg-prepared_${SPLIT}_s${SEED}${HO_PART}.pkl"
fi

SHUFFLE_EXTRA=()
if [[ "${SHUFFLE_RELATIONS:-0}" == "1" ]]; then
  SHUFFLE_EXTRA=(--shuffle-relations)
fi

HOLDOUT_EXPORT=()
if [[ -n "${HELD_OUT_RELATIONS:-}" ]]; then
  read -r -a _hor <<< "${HELD_OUT_RELATIONS}"
  if [[ ${#_hor[@]} -gt 0 ]]; then
    HOLDOUT_EXPORT=(--held-out-relations "${_hor[@]}")
  fi
fi

EXPORT_CMD=(
  python -m kgml_new.scripts.export_prepared_link_prediction
  --input "${INPUT}"
  --input-format auto
  --split-protocol "${SPLIT}"
  --seed "${SEED}"
  --in-dim "${IN_DIM}"
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
EXPORT_CMD+=("${SHUFFLE_EXTRA[@]}" "${HOLDOUT_EXPORT[@]}")

echo "=== drkg_build_prepared_cache host=$(hostname) ==="
echo "Input: ${INPUT}"
echo "Output: ${PREPARED_CACHE}"

if [[ -f "${PREPARED_CACHE}" ]] && [[ "${FORCE_REBUILD_PREPARED_CACHE:-0}" != "1" ]]; then
  echo "Cache already exists; skipping (set FORCE_REBUILD_PREPARED_CACHE=1 to rebuild)."
  exit 0
fi

set -x
"${EXPORT_CMD[@]}"
set +x

echo "Wrote ${PREPARED_CACHE}"
