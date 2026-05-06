#!/bin/bash
# One-hot edge-aware SAGE ablation: PrimeKG and FB15k-237, node- and edge-disjoint splits.
# No semantic relation cache or API keys.
#
#   sbatch scripts/slurm/run_edge_aware_sage_onehot_primekg_fb15k_array.slurm
#
# Requires prepared caches (CPU) from:
#   sbatch scripts/slurm/primekg_fb15k_build_prepared_caches.slurm
#   -> cache/{dataset}-prepared_{edge|node}_s{SEED}_d{IN_DIM}.pkl
#
#SBATCH --job-name=kgml-eas1h-pk-fb15k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/slurm/kgml-eas1h-pk-fb15k-%A_%a.out
#SBATCH --error=logs/slurm/kgml-eas1h-pk-fb15k-%A_%a.err

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
EDGE_DIM="${EDGE_DIM:-128}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
EPOCHS="${EPOCHS:-100}"

PRIMEKG_INPUT="${PRIMEKG_INPUT:-${REPO}/data/primekg.tsv}"
FB15K237_INPUT="${FB15K237_INPUT:-${REPO}/data/fb15k-237/fb15k-237.tsv}"

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

PREPARED_CACHE="${REPO}/cache/${DATASET}-prepared_${SPLIT}_s${SEED}_d${IN_DIM}.pkl"
if [[ ! -f "${PREPARED_CACHE}" ]]; then
  echo "ERROR: Prepared cache missing: ${PREPARED_CACHE}" >&2
  echo "Run CPU cache build first:" >&2
  echo "  sbatch scripts/slurm/primekg_fb15k_build_prepared_caches.slurm" >&2
  exit 1
fi

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

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
RUN_NAME="edge_aware_sage_onehot_concat"
OUT="${REPO}/results/slurm/${DATASET}_${SPLIT}_${RUN_NAME}_${JOB_TAG}.json"
LOG="${REPO}/logs/slurm/run-${DATASET}-${SPLIT}-${RUN_NAME}-${JOB_TAG}.log"

echo "=== run_edge_aware_sage_onehot_primekg_fb15k host=$(hostname) task=${TASK} dataset=${DATASET} split=${SPLIT} ==="
echo "Prepared cache: ${PREPARED_CACHE}"

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method edge_aware_sage_onehot \
  --task pairwise_link_prediction \
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
  --prepared-dataset-cache "${PREPARED_CACHE}" \
  --epochs "${EPOCHS}" \
  --edge-relation-mode concat \
  --edge-dim "${EDGE_DIM}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}"
set +x

echo "Wrote ${OUT}"
