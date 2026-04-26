#!/bin/bash
# GPU array sweep over:
#   datasets: primekg, fb15k237
#   splits: edge, node
#   methods: node2vec, baseline_gcn, baseline_sage, edge_aware_sage+concat(openai|gemini|e5)
#
# Requires prepared caches built beforehand (CPU):
#   cache/{dataset}-prepared_{split}_s{SEED}_d{IN_DIM}.pkl
#
#SBATCH --job-name=kgml-primekg-fb15k-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-23%6
#SBATCH --output=logs/slurm/kgml-primekg-fb15k-sweep-%A_%a.out
#SBATCH --error=logs/slurm/kgml-primekg-fb15k-sweep-%A_%a.err

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
PRIMEKG_OPENAI_REL_CACHE="${PRIMEKG_OPENAI_REL_CACHE:-${OPENAI_REL_CACHE:-cache/primekg-relations-openai.pt}}"
PRIMEKG_GEMINI_REL_CACHE="${PRIMEKG_GEMINI_REL_CACHE:-${GEMINI_REL_CACHE:-cache/primekg-relations-gemini.pt}}"
PRIMEKG_E5_REL_CACHE="${PRIMEKG_E5_REL_CACHE:-${E5_REL_CACHE:-cache/primekg-relations-e5.pt}}"
FB15K237_OPENAI_REL_CACHE="${FB15K237_OPENAI_REL_CACHE:-${OPENAI_REL_CACHE:-cache/fb15k237-relations-openai.pt}}"
FB15K237_GEMINI_REL_CACHE="${FB15K237_GEMINI_REL_CACHE:-${GEMINI_REL_CACHE:-cache/fb15k237-relations-gemini.pt}}"
FB15K237_E5_REL_CACHE="${FB15K237_E5_REL_CACHE:-${E5_REL_CACHE:-cache/fb15k237-relations-e5.pt}}"

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
if [[ -z "${TASK}" ]] || [[ "${TASK}" -lt 0 ]] || [[ "${TASK}" -gt 23 ]]; then
  echo "ERROR: Invalid SLURM_ARRAY_TASK_ID=${TASK}" >&2
  exit 1
fi

DATASETS=(primekg fb15k237)
SPLITS=(edge node)
RUN_SPECS=(
  node2vec
  baseline_gcn
  baseline_sage
  edge_aware_sage_openai
  edge_aware_sage_gemini
  edge_aware_sage_e5
)

SPEC_IDX=$((TASK % 6))
SPLIT_IDX=$(((TASK / 6) % 2))
DATASET_IDX=$((TASK / 12))

DATASET="${DATASETS[$DATASET_IDX]}"
SPLIT="${SPLITS[$SPLIT_IDX]}"
RUN_SPEC="${RUN_SPECS[$SPEC_IDX]}"

if [[ "${DATASET}" == "primekg" ]]; then
  INPUT="${PRIMEKG_INPUT}"
else
  INPUT="${FB15K237_INPUT}"
fi

REL_CACHE_OPENAI=""
REL_CACHE_GEMINI=""
REL_CACHE_E5=""
if [[ "${DATASET}" == "primekg" ]]; then
  REL_CACHE_OPENAI="${PRIMEKG_OPENAI_REL_CACHE}"
  REL_CACHE_GEMINI="${PRIMEKG_GEMINI_REL_CACHE}"
  REL_CACHE_E5="${PRIMEKG_E5_REL_CACHE}"
else
  REL_CACHE_OPENAI="${FB15K237_OPENAI_REL_CACHE}"
  REL_CACHE_GEMINI="${FB15K237_GEMINI_REL_CACHE}"
  REL_CACHE_E5="${FB15K237_E5_REL_CACHE}"
fi

if [[ -n "${REL_CACHE_OPENAI}" ]] && [[ "${REL_CACHE_OPENAI}" != /* ]]; then
  REL_CACHE_OPENAI="${REPO}/${REL_CACHE_OPENAI}"
fi
if [[ -n "${REL_CACHE_GEMINI}" ]] && [[ "${REL_CACHE_GEMINI}" != /* ]]; then
  REL_CACHE_GEMINI="${REPO}/${REL_CACHE_GEMINI}"
fi
if [[ -n "${REL_CACHE_E5}" ]] && [[ "${REL_CACHE_E5}" != /* ]]; then
  REL_CACHE_E5="${REPO}/${REL_CACHE_E5}"
fi

PREPARED_CACHE="${REPO}/cache/${DATASET}-prepared_${SPLIT}_s${SEED}_d${IN_DIM}.pkl"
if [[ ! -f "${PREPARED_CACHE}" ]]; then
  echo "ERROR: Prepared cache missing: ${PREPARED_CACHE}" >&2
  echo "Run CPU cache build first:" >&2
  echo "  sbatch scripts/slurm/primekg_fb15k_build_prepared_caches.slurm" >&2
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

METHOD_ARGS=()
RUN_NAME=""

case "${RUN_SPEC}" in
  node2vec)
    RUN_NAME="node2vec"
    METHOD_ARGS=(--method node2vec)
    # Prepared caches (--prepared-dataset-cache) contain the full graph built at cache time.
    # Do not pass --max-edges here: it only applies to CSV loads and incorrectly broke cache
    # validation (meta mismatch). For a CSV-only PrimeKG node2vec cap, set MAX_EDGES and run
    # without a prepared cache in a dedicated script.
    ;;
  baseline_gcn)
    RUN_NAME="baseline_gcn"
    METHOD_ARGS=(--method baseline_gcn)
    ;;
  baseline_sage)
    RUN_NAME="baseline_sage"
    METHOD_ARGS=(--method baseline_sage)
    ;;
  edge_aware_sage_openai)
    if [[ -z "${REL_CACHE_OPENAI}" ]] || [[ ! -f "${REL_CACHE_OPENAI}" ]]; then
      echo "ERROR: Existing OpenAI relation cache required for ${DATASET}." >&2
      echo "Set PRIMEKG_OPENAI_REL_CACHE/FB15K237_OPENAI_REL_CACHE (or OPENAI_REL_CACHE)." >&2
      exit 1
    fi
    RUN_NAME="edge_aware_sage_concat_openai"
    METHOD_ARGS=(
      --method edge_aware_sage
      --semantic
      --strict-semantic
      --embedding-model openai
      --edge-relation-mode concat
      --edge-dim "${EDGE_DIM}"
      --semantic-cache "${REL_CACHE_OPENAI}"
    )
    ;;
  edge_aware_sage_gemini)
    if [[ -z "${REL_CACHE_GEMINI}" ]] || [[ ! -f "${REL_CACHE_GEMINI}" ]]; then
      echo "ERROR: Existing Gemini relation cache required for ${DATASET}." >&2
      echo "Set PRIMEKG_GEMINI_REL_CACHE/FB15K237_GEMINI_REL_CACHE (or GEMINI_REL_CACHE)." >&2
      exit 1
    fi
    RUN_NAME="edge_aware_sage_concat_gemini"
    METHOD_ARGS=(
      --method edge_aware_sage
      --semantic
      --strict-semantic
      --embedding-model gemini
      --edge-relation-mode concat
      --edge-dim "${EDGE_DIM}"
      --semantic-cache "${REL_CACHE_GEMINI}"
    )
    ;;
  edge_aware_sage_e5)
    if [[ -z "${REL_CACHE_E5}" ]] || [[ ! -f "${REL_CACHE_E5}" ]]; then
      echo "ERROR: Existing E5 relation cache required for ${DATASET}." >&2
      echo "Set PRIMEKG_E5_REL_CACHE/FB15K237_E5_REL_CACHE (or E5_REL_CACHE)." >&2
      exit 1
    fi
    RUN_NAME="edge_aware_sage_concat_e5"
    METHOD_ARGS=(
      --method edge_aware_sage
      --semantic
      --strict-semantic
      --embedding-model e5
      --edge-relation-mode concat
      --edge-dim "${EDGE_DIM}"
      --semantic-cache "${REL_CACHE_E5}"
    )
    ;;
  *)
    echo "ERROR: Unknown run spec ${RUN_SPEC}" >&2
    exit 1
    ;;
esac

if [[ "${EDGE_DIM}" != "128" ]]; then
  echo "ERROR: EDGE_DIM must remain 128 for edge-aware concat runs (got ${EDGE_DIM})." >&2
  exit 1
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT="${REPO}/results/slurm/${DATASET}_${SPLIT}_${RUN_NAME}_${JOB_TAG}.json"
LOG="${REPO}/logs/slurm/run-${DATASET}-${SPLIT}-${RUN_NAME}-${JOB_TAG}.log"

echo "=== primekg_fb15k_full_sweep_array host=$(hostname) task=${TASK} dataset=${DATASET} split=${SPLIT} run=${RUN_NAME} ==="
echo "Using prepared cache: ${PREPARED_CACHE}"

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
  "${NEG_EXTRA[@]}" \
  "${MAX_EXTRA[@]}" \
  "${SHUFFLE_EXTRA[@]}" \
  --prepared-dataset-cache "${PREPARED_CACHE}" \
  --epochs "${EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}" \
  "${METHOD_ARGS[@]}"
set +x

echo "Wrote ${OUT}"
