#!/bin/bash
# Build prepared link-prediction caches on CPU for:
#   - PrimeKG
#   - FB15k-237
# and for both splits:
#   - edge
#   - node
#
# This job is CPU-only and should run before GPU training arrays.
#
#SBATCH --job-name=kgml-prep-primekg-fb15k
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/kgml-prep-primekg-fb15k-%j.out
#SBATCH --error=logs/slurm/kgml-prep-primekg-fb15k-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"
mkdir -p logs/slurm cache

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in ${REPO}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-128}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

PRIMEKG_INPUT="${PRIMEKG_INPUT:-${REPO}/data/primekg.tsv}"
FB15K237_INPUT="${FB15K237_INPUT:-${REPO}/data/fb15k-237/fb15k-237.tsv}"

if [[ "${PRIMEKG_INPUT}" != /* ]]; then
  PRIMEKG_INPUT="${REPO}/${PRIMEKG_INPUT}"
fi
if [[ "${FB15K237_INPUT}" != /* ]]; then
  FB15K237_INPUT="${REPO}/${FB15K237_INPUT}"
fi
if [[ ! -f "${PRIMEKG_INPUT}" ]]; then
  echo "ERROR: PrimeKG input not found: ${PRIMEKG_INPUT}" >&2
  exit 1
fi
if [[ ! -f "${FB15K237_INPUT}" ]]; then
  echo "ERROR: FB15k-237 input not found: ${FB15K237_INPUT}" >&2
  exit 1
fi

MAX_EXTRA=()
if [[ -n "${MAX_EDGES:-}" ]]; then
  MAX_EXTRA=(--max-edges "${MAX_EDGES}")
fi

NEG_EXTRA=()
if [[ -n "${NEGATIVE_SAMPLING_MODE:-}" ]]; then
  NEG_EXTRA=(--negative-sampling-mode "${NEGATIVE_SAMPLING_MODE}")
fi

SHUFFLE_EXTRA=()
if [[ "${SHUFFLE_RELATIONS:-0}" == "1" ]]; then
  SHUFFLE_EXTRA=(--shuffle-relations)
fi

echo "=== primekg_fb15k_build_prepared_caches host=$(hostname) ==="
echo "SEED=${SEED} IN_DIM=${IN_DIM} DECODER=${DECODER} NEGATIVES_PER_POS=${NEGATIVES_PER_POS}"

for DATASET in primekg fb15k237; do
  if [[ "${DATASET}" == "primekg" ]]; then
    INPUT="${PRIMEKG_INPUT}"
  else
    INPUT="${FB15K237_INPUT}"
  fi

  for SPLIT in edge node; do
    PREPARED_CACHE="${REPO}/cache/${DATASET}-prepared_${SPLIT}_s${SEED}_d${IN_DIM}.pkl"
    echo "--- dataset=${DATASET} split=${SPLIT} cache=${PREPARED_CACHE} ---"

    if [[ -f "${PREPARED_CACHE}" ]] && [[ "${FORCE_REBUILD_PREPARED_CACHE:-0}" != "1" ]]; then
      echo "Cache exists; skipping (set FORCE_REBUILD_PREPARED_CACHE=1 to rebuild)."
      continue
    fi

    set -x
    python -m kgml_new.scripts.export_prepared_link_prediction \
      --input "${INPUT}" \
      --input-format auto \
      --split-protocol "${SPLIT}" \
      --seed "${SEED}" \
      --in-dim "${IN_DIM}" \
      --negatives-per-pos "${NEGATIVES_PER_POS}" \
      --decoder "${DECODER}" \
      --val-ratio "${VAL_RATIO}" \
      --test-ratio "${TEST_RATIO}" \
      "${MAX_EXTRA[@]}" \
      "${NEG_EXTRA[@]}" \
      "${SHUFFLE_EXTRA[@]}" \
      --output "${PREPARED_CACHE}"
    set +x
  done
done

echo "=== primekg_fb15k_build_prepared_caches done ==="
