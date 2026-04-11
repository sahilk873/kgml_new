#!/bin/bash
# Node-split DRKG sweep: baseline vs edge-aware, semantic vs random, dim 128/256.
#
# Matrix (36 tasks):
#   - baseline_sage x {mean,max} x {128,256} = 4
#   - edge_aware_sage x {concat,gated,basis_mixture,film} x {mean,max} x {128,256} x {sapbert,random} = 32
#
# Output pattern:
#   results/slurm/drkg_node_semrand_${NAME}_${JOB_TAG}.json
#
# Defaults follow existing full runs unless overridden:
#   SPLIT=node, EPOCHS=100, SEED=42, DECODER=dot, NEGATIVES_PER_POS=20
#
# Prepared cache: per --in-dim, default path is
#   cache/drkg-prepared_${SPLIT}_s${SEED}_d${DIM}.pkl
# This GPU job does NOT build caches (CPU-heavy). Build first:
#   sbatch scripts/slurm/drkg_build_prepared_node_dim_caches.slurm
# or use ./scripts/slurm/submit_drkg_node_semantic_random_dim_sweep.sh (CPU build + GPU afterok).
# Override single path for all tasks: PREPARED_CACHE=/path/to.pkl (must match each task's in_dim).
#
#SBATCH --job-name=drkg-node-semrand
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-35%8
#SBATCH --output=logs/slurm/drkg-node-semrand-%A_%a.out
#SBATCH --error=logs/slurm/drkg-node-semrand-%A_%a.err

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

SPLIT="${SPLIT:-node}"
EPOCHS="${EPOCHS:-100}"
SEED="${SEED:-42}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
USE_PREPARED_DATASET_CACHE="${USE_PREPARED_DATASET_CACHE:-1}"
INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"

SAPBERT_REL_CACHE="${SAPBERT_REL_CACHE:-cache/drkg-relations-sapbert.pt}"
RANDOM_REL_CACHE="${RANDOM_REL_CACHE:-cache/drkg-relations-random.pt}"
NUM_RELATION_BASES="${NUM_RELATION_BASES:-4}"
SEMANTIC_ALIGNMENT_LAMBDA="${SEMANTIC_ALIGNMENT_LAMBDA:-0.0}"

if [[ "${INPUT}" != /* ]]; then
  INPUT="${REPO}/${INPUT}"
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input KG file not found: ${INPUT}" >&2
  exit 1
fi

if [[ "${SAPBERT_REL_CACHE}" != /* ]]; then
  SAPBERT_REL_CACHE="${REPO}/${SAPBERT_REL_CACHE}"
fi
if [[ "${RANDOM_REL_CACHE}" != /* ]]; then
  RANDOM_REL_CACHE="${REPO}/${RANDOM_REL_CACHE}"
fi

MAX_EXTRA=()
if [[ -n "${MAX_EDGES:-}" ]]; then
  MAX_EXTRA=(--max-edges "${MAX_EDGES}")
fi

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
  elif [[ -n "${DIVERSITY_BUCKETS_CACHE:-}" ]]; then
    echo "ERROR: DIVERSITY_BUCKETS_CACHE not found: ${DIV_CAND}" >&2
    exit 1
  fi
fi

NEG_EXTRA=()
if [[ -n "${NEGATIVE_SAMPLING_MODE:-}" ]]; then
  NEG_EXTRA=(--negative-sampling-mode "${NEGATIVE_SAMPLING_MODE}")
fi

TASK="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK}" ]] || [[ "${TASK}" -lt 0 ]] || [[ "${TASK}" -gt 35 ]]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK}" >&2
  exit 1
fi

METHOD=""
MODE=""
AGGR=""
DIM=""
CONTROL=""
CONTROL_EXTRA=()
BASIS_EXTRA=()

if [[ "${TASK}" -lt 4 ]]; then
  METHOD="baseline_sage"
  BASE_DIMS=(128 256)
  BASE_AGGRS=(mean max)
  DIM="${BASE_DIMS[$((TASK / 2))]}"
  AGGR="${BASE_AGGRS[$((TASK % 2))]}"
  CONTROL="baseline"
  NAME="baseline_sage_${AGGR}_d${DIM}"
else
  METHOD="edge_aware_sage"
  MODES=(concat gated basis_mixture film)
  AGGRS=(mean max)
  DIMS=(128 256)
  CONTROLS=(sapbert random)

  IDX=$((TASK - 4))
  CONTROL="${CONTROLS[$((IDX % 2))]}"
  IDX=$((IDX / 2))
  DIM="${DIMS[$((IDX % 2))]}"
  IDX=$((IDX / 2))
  AGGR="${AGGRS[$((IDX % 2))]}"
  IDX=$((IDX / 2))
  MODE="${MODES[$IDX]}"

  if [[ "${CONTROL}" == "sapbert" ]]; then
    if [[ ! -f "${SAPBERT_REL_CACHE}" ]]; then
      echo "ERROR: SAPBERT_REL_CACHE not found: ${SAPBERT_REL_CACHE}" >&2
      exit 1
    fi
    CONTROL_EXTRA=(--semantic-cache "${SAPBERT_REL_CACHE}" --embedding-model sapbert)
  else
    if [[ ! -f "${RANDOM_REL_CACHE}" ]]; then
      echo "ERROR: RANDOM_REL_CACHE not found: ${RANDOM_REL_CACHE}" >&2
      exit 1
    fi
    CONTROL_EXTRA=(--semantic-cache "${RANDOM_REL_CACHE}" --embedding-model random)
  fi

  if [[ "${MODE}" == "basis_mixture" ]]; then
    BASIS_EXTRA=(
      --num-relation-bases "${NUM_RELATION_BASES}"
      --semantic-alignment-lambda "${SEMANTIC_ALIGNMENT_LAMBDA}"
    )
  fi

  NAME="edge_aware_${MODE}_${AGGR}_d${DIM}_${CONTROL}"
fi

PREPARED_EXTRA=()
if [[ "${USE_PREPARED_DATASET_CACHE}" == "1" ]]; then
  if [[ -n "${PREPARED_CACHE:-}" ]]; then
    if [[ "${PREPARED_CACHE}" != /* ]]; then
      PREPARED_CACHE="${REPO}/${PREPARED_CACHE}"
    fi
  else
    PREPARED_CACHE="${REPO}/cache/drkg-prepared_${SPLIT}_s${SEED}_d${DIM}.pkl"
  fi
  if [[ ! -f "${PREPARED_CACHE}" ]]; then
    echo "ERROR: Prepared dataset cache not found: ${PREPARED_CACHE}" >&2
    echo "  GPU jobs do not run export (CPU-heavy). Build on CPU first, e.g.:" >&2
    echo "    sbatch --partition=\${BUILD_PARTITION:-batch} scripts/slurm/drkg_build_prepared_node_dim_caches.slurm" >&2
    echo "  Or submit CPU build + GPU sweep: ./scripts/slurm/submit_drkg_node_semantic_random_dim_sweep.sh" >&2
    exit 1
  fi
  PREPARED_EXTRA=(--prepared-dataset-cache "${PREPARED_CACHE}")
fi

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False on this GPU node." >&2
  exit 1
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
LOG="${REPO}/logs/slurm/run-drkg-node-semrand-${NAME}-${JOB_TAG}.log"
OUT="${REPO}/results/slurm/drkg_node_semrand_${NAME}_${JOB_TAG}.json"

echo "=== drkg_node_semantic_random_dim_sweep task=${TASK} split=${SPLIT} method=${METHOD} mode=${MODE:-n/a} aggr=${AGGR} dim=${DIM} control=${CONTROL} host=$(hostname) ==="

RUN_EXTRA=()
if [[ "${METHOD}" == "baseline_sage" ]]; then
  RUN_EXTRA=(--method baseline_sage --neighbor-aggr "${AGGR}")
else
  RUN_EXTRA=(
    --method edge_aware_sage
    --edge-relation-mode "${MODE}"
    --neighbor-aggr "${AGGR}"
    "${CONTROL_EXTRA[@]}"
    "${BASIS_EXTRA[@]}"
  )
fi

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --input "${INPUT}" \
  --input-format auto \
  --split-protocol "${SPLIT}" \
  --seed "${SEED}" \
  --in-dim "${DIM}" \
  --decoder "${DECODER}" \
  --negatives-per-pos "${NEGATIVES_PER_POS}" \
  "${NEG_EXTRA[@]}" \
  "${MAX_EXTRA[@]}" \
  "${DIV_EXTRA[@]}" \
  "${PREPARED_EXTRA[@]}" \
  --epochs "${EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}" \
  "${RUN_EXTRA[@]}"
set +x

echo "Wrote ${OUT}"
