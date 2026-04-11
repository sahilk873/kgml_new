#!/bin/bash
# Node-split semantic backfill for missing edge-aware GraphSAGE runs.
#
# 4-task array (all neighbor_aggr=max):
#   0 -> film
#   1 -> concat
#   2 -> gated
#   3 -> basis_mixture
#
# Uses existing prepared node cache and SapBERT relation cache.
#
#SBATCH --job-name=drkg-node-sem-backfill
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/slurm/drkg-node-sem-backfill-%A_%a.out
#SBATCH --error=logs/slurm/drkg-node-sem-backfill-%A_%a.err

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
SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-64}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
USE_PREPARED_DATASET_CACHE="${USE_PREPARED_DATASET_CACHE:-1}"
SPLIT="node"
INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"
REL_CACHE="${REL_CACHE:-cache/drkg-relations-sapbert.pt}"
NUM_RELATION_BASES="${NUM_RELATION_BASES:-4}"
SEMANTIC_ALIGNMENT_LAMBDA="${SEMANTIC_ALIGNMENT_LAMBDA:-0.0}"

if [[ "${INPUT}" != /* ]]; then
  INPUT="${REPO}/${INPUT}"
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input KG file not found: ${INPUT}" >&2
  exit 1
fi

if [[ "${REL_CACHE}" != /* ]]; then
  REL_CACHE="${REPO}/${REL_CACHE}"
fi
if [[ ! -f "${REL_CACHE}" ]]; then
  echo "ERROR: REL_CACHE not found: ${REL_CACHE}" >&2
  exit 1
fi

PREPARED_EXTRA=()
if [[ "${USE_PREPARED_DATASET_CACHE}" == "1" ]]; then
  if [[ -n "${PREPARED_CACHE:-}" ]]; then
    if [[ "${PREPARED_CACHE}" != /* ]]; then
      PREPARED_CACHE="${REPO}/${PREPARED_CACHE}"
    fi
  else
    PREPARED_CACHE="${REPO}/cache/drkg-prepared_node_s${SEED}.pkl"
  fi
  if [[ ! -f "${PREPARED_CACHE}" ]]; then
    echo "ERROR: Prepared dataset cache not found: ${PREPARED_CACHE}" >&2
    exit 1
  fi
  PREPARED_EXTRA=(--prepared-dataset-cache "${PREPARED_CACHE}")
fi

MAX_EXTRA=()
if [[ -n "${MAX_EDGES:-}" ]]; then
  MAX_EXTRA=(--max-edges "${MAX_EDGES}")
fi

NEG_EXTRA=()
if [[ -n "${NEGATIVE_SAMPLING_MODE:-}" ]]; then
  NEG_EXTRA=(--negative-sampling-mode "${NEGATIVE_SAMPLING_MODE}")
fi

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False on this GPU node." >&2
  exit 1
fi

case "${SLURM_ARRAY_TASK_ID:-}" in
  0)
    MODE="film"
    EXTRA=()
    ;;
  1)
    MODE="concat"
    EXTRA=()
    ;;
  2)
    MODE="gated"
    EXTRA=()
    ;;
  3)
    MODE="basis_mixture"
    EXTRA=(
      --num-relation-bases "${NUM_RELATION_BASES}"
      --semantic-alignment-lambda "${SEMANTIC_ALIGNMENT_LAMBDA}"
    )
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" >&2
    exit 1
    ;;
esac

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
LOG="${REPO}/logs/slurm/run-drkg-node-sem-${MODE}-${JOB_TAG}.log"
OUT="${REPO}/results/slurm/drkg_node_semantic_${MODE}_pool_${JOB_TAG}.json"

echo "=== drkg_node_semantic_backfill task=${SLURM_ARRAY_TASK_ID:-?} mode=${MODE} host=$(hostname) ==="

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method edge_aware_sage \
  --edge-relation-mode "${MODE}" \
  --neighbor-aggr max \
  --semantic-cache "${REL_CACHE}" \
  --embedding-model sapbert \
  "${EXTRA[@]}" \
  --input "${INPUT}" \
  --input-format auto \
  --split-protocol "${SPLIT}" \
  --seed "${SEED}" \
  --in-dim "${IN_DIM}" \
  --decoder "${DECODER}" \
  --negatives-per-pos "${NEGATIVES_PER_POS}" \
  "${NEG_EXTRA[@]}" \
  "${MAX_EXTRA[@]}" \
  "${PREPARED_EXTRA[@]}" \
  --epochs "${EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}"
set +x

echo "Wrote ${OUT}"
