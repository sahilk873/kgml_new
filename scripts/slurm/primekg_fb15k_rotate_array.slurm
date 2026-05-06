#!/bin/bash
# GPU array: RotatE on PrimeKG and FB15k-237 with edge / node split_protocol.
#
# Requires prepared caches (CPU) from primekg_fb15k_build_prepared_caches.slurm:
#   ${CACHE_DIR:-$REPO/cache}/{dataset}-prepared_{split}_s{SEED}_d{IN_DIM}.pkl
#
# Same layout as primekg_fb15k_full_sweep_array.slurm. Optional overrides:
#   CACHE_DIR          — directory holding prepared *.pkl files (default: $REPO/cache).
#   PREPARED_CACHE     — use this exact pickle (absolute or repo-relative); skips
#                        default naming. For array jobs, leave unset so each task
#                        picks its own file from CACHE_DIR.
# If the default path is missing, the script tries:
#   {dataset}-prepared_{split}_s{SEED}.pkl (legacy, no _d{IN_DIM}),
#   then a single glob match for {dataset}-prepared_{split}_s{SEED}_*.pkl.
# When the resolved path ends with _d{NNN}.pkl, IN_DIM is aligned to NNN so CLI
# meta matches the pickle (override a wrong IN_DIM default).
#
# Array tasks 0-3:
#   0: primekg + edge
#   1: primekg + node
#   2: fb15k237 + edge
#   3: fb15k237 + node
#
#SBATCH --job-name=kgml-rotate-pk-fb15k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/slurm/kgml-rotate-pk-fb15k-%A_%a.out
#SBATCH --error=logs/slurm/kgml-rotate-pk-fb15k-%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"
mkdir -p logs/slurm results/slurm cache

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in ${REPO}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False on this node." >&2
  exit 1
fi

SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-128}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
EPOCHS="${EPOCHS:-100}"

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

TASK="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK}" ]] || [[ "${TASK}" -lt 0 ]] || [[ "${TASK}" -gt 3 ]]; then
  echo "ERROR: Invalid SLURM_ARRAY_TASK_ID=${TASK} (expected 0-3)" >&2
  exit 1
fi

DATASETS=(primekg fb15k237)
SPLITS=(edge node)

DATASET_IDX=$((TASK / 2))
SPLIT_IDX=$((TASK % 2))
DATASET="${DATASETS[$DATASET_IDX]}"
SPLIT="${SPLITS[$SPLIT_IDX]}"

if [[ "${DATASET}" == "primekg" ]]; then
  INPUT="${PRIMEKG_INPUT}"
else
  INPUT="${FB15K237_INPUT}"
fi

CACHE_DIR="${CACHE_DIR:-${REPO}/cache}"
if [[ "${CACHE_DIR}" != /* ]]; then
  CACHE_DIR="${REPO}/${CACHE_DIR}"
fi

PREPARED_OVERRIDE="${PREPARED_CACHE:-}"
if [[ -n "${PREPARED_OVERRIDE}" ]]; then
  if [[ "${PREPARED_OVERRIDE}" != /* ]]; then
    PREPARED_OVERRIDE="${REPO}/${PREPARED_OVERRIDE}"
  fi
  PREPARED_CACHE="${PREPARED_OVERRIDE}"
else
  PREPARED_CACHE="${CACHE_DIR}/${DATASET}-prepared_${SPLIT}_s${SEED}_d${IN_DIM}.pkl"
  if [[ ! -f "${PREPARED_CACHE}" ]]; then
    LEGACY="${CACHE_DIR}/${DATASET}-prepared_${SPLIT}_s${SEED}.pkl"
    if [[ -f "${LEGACY}" ]]; then
      echo "NOTE: Using legacy prepared cache (no _d{IN_DIM} in filename): ${LEGACY}" >&2
      PREPARED_CACHE="${LEGACY}"
    fi
  fi
  if [[ ! -f "${PREPARED_CACHE}" ]]; then
    shopt -s nullglob
    GLOB_CAND=( "${CACHE_DIR}/${DATASET}-prepared_${SPLIT}_s${SEED}"_*.pkl )
    shopt -u nullglob
    if [[ ${#GLOB_CAND[@]} -eq 1 ]] && [[ -f "${GLOB_CAND[0]}" ]]; then
      echo "NOTE: Auto-selected sole matching prepared cache: ${GLOB_CAND[0]}" >&2
      PREPARED_CACHE="${GLOB_CAND[0]}"
    elif [[ ${#GLOB_CAND[@]} -gt 1 ]]; then
      echo "ERROR: Multiple prepared caches for ${DATASET} split=${SPLIT} seed=${SEED} under ${CACHE_DIR}:" >&2
      printf '  %s\n' "${GLOB_CAND[@]}" >&2
      echo "Disambiguate by setting IN_DIM (expects ..._d\${IN_DIM}.pkl), CACHE_DIR, or PREPARED_CACHE=/path/to/one.pkl" >&2
      exit 1
    fi
  fi
fi

if [[ ! -f "${PREPARED_CACHE}" ]]; then
  echo "ERROR: Prepared cache missing: ${PREPARED_CACHE}" >&2
  echo "Expected default: ${CACHE_DIR}/${DATASET}-prepared_${SPLIT}_s${SEED}_d${IN_DIM}.pkl" >&2
  echo "Run CPU cache build first:" >&2
  echo "  sbatch scripts/slurm/primekg_fb15k_build_prepared_caches.slurm" >&2
  exit 1
fi

if [[ "${PREPARED_CACHE}" =~ _d([0-9]+)\.pkl$ ]]; then
  _DIM_FROM_NAME="${BASH_REMATCH[1]}"
  if [[ "${IN_DIM}" != "${_DIM_FROM_NAME}" ]]; then
    echo "NOTE: Aligning IN_DIM ${IN_DIM} -> ${_DIM_FROM_NAME} (from prepared cache filename for meta validation)." >&2
    IN_DIM="${_DIM_FROM_NAME}"
  fi
fi

# Prepared-cache meta includes graph_csv_spec; CLI CSV columns must match how each pickle was built.
# Default matches caches produced alongside primekg_fb15k_build_prepared_caches (relation + extra cols).
CSV_SPEC_EXTRA=(--relation-col "${CSV_RELATION_COL:-relation}")
if [[ "${CSV_EDGE_ATTR_COLS:-}" == "none" ]]; then
  :
elif [[ -n "${CSV_EDGE_ATTR_COLS:-}" ]]; then
  IFS=',' read -ra _eat <<< "${CSV_EDGE_ATTR_COLS}"
  for c in "${_eat[@]}"; do
    c="${c// /}"
    [[ -n "${c}" ]] && CSV_SPEC_EXTRA+=(--edge-attr-col "${c}")
  done
else
  for c in display_relation x_source y_source; do
    CSV_SPEC_EXTRA+=(--edge-attr-col "${c}")
  done
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

ROTATE_EXTRA=()
if [[ -n "${ROTATE_EMBEDDING_DIM:-}" ]]; then
  ROTATE_EXTRA+=(--rotate-embedding-dim "${ROTATE_EMBEDDING_DIM}")
fi
if [[ -n "${ROTATE_GAMMA:-}" ]]; then
  ROTATE_EXTRA+=(--rotate-gamma "${ROTATE_GAMMA}")
fi
if [[ -n "${ROTATE_BATCH_SIZE:-}" ]]; then
  ROTATE_EXTRA+=(--rotate-batch-size "${ROTATE_BATCH_SIZE}")
fi
if [[ -n "${ROTATE_NEG_SAMPLES:-}" ]]; then
  ROTATE_EXTRA+=(--rotate-neg-samples "${ROTATE_NEG_SAMPLES}")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
  ROTATE_EXTRA+=(--learning-rate "${LEARNING_RATE}")
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT="${REPO}/results/slurm/${DATASET}_${SPLIT}_rotate_${JOB_TAG}.json"
LOG="${REPO}/logs/slurm/run-${DATASET}-${SPLIT}-rotate-${JOB_TAG}.log"

echo "=== primekg_fb15k_rotate_array host=$(hostname) task=${TASK} dataset=${DATASET} split=${SPLIT} ==="
echo "Using prepared cache: ${PREPARED_CACHE}"

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method rotate \
  --input "${INPUT}" \
  "${CSV_SPEC_EXTRA[@]}" \
  --input-format auto \
  --split-protocol "${SPLIT}" \
  --seed "${SEED}" \
  --in-dim "${IN_DIM}" \
  --decoder "${DECODER}" \
  --negatives-per-pos "${NEGATIVES_PER_POS}" \
  "${NEG_EXTRA[@]}" \
  "${MAX_EXTRA[@]}" \
  "${SHUFFLE_EXTRA[@]}" \
  "${ROTATE_EXTRA[@]}" \
  --prepared-dataset-cache "${PREPARED_CACHE}" \
  --epochs "${EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}"
set +x

echo "Wrote ${OUT}"
