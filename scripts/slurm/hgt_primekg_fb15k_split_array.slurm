#!/bin/bash
# Train HGT link prediction on PrimeKG and FB15k-237 with edge vs node splits.
#
# GPU array tasks 0–3:
#   Task  PrimeKG/Fb15k × edge/node (see TASK mapping below).
#
# PrimeKG defaults:
#   PRIMEKG_INPUT (default data/primekg.tsv), HGT_RELATION_PRIMEKG (default indication).
#
# FB15k-237 (headerless TSV: source TAB relation TAB target):
#   FB15K237_INPUT (default data/fb15k-237/fb15k-237.tsv).
#   HGT_RELATION_FB15K237 must be set to a relation string that appears in column 2
#   (example: pick a frequent predicate via
#    cut -f2 data/fb15k-237/fb15k-237.tsv | sort | uniq -c | sort -nr | head).
#
# Optional disambiguation when multiple typed edges share a relation name:
#   HGT_SOURCE_NODE_TYPE, HGT_TARGET_NODE_TYPE  → passed as --source-node-type / --target-node-type.
#
#SBATCH --job-name=kgml-hgt-pk-fb15k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/slurm/kgml-hgt-primekg-fb15k-%A_%a.out
#SBATCH --error=logs/slurm/kgml-hgt-primekg-fb15k-%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
mkdir -p logs/slurm results/slurm

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in ${REPO}" >&2
  exit 1
fi
PYTHON="${PYTHON:-${REPO}/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} missing or not executable." >&2
  echo "       Repair: cd ${REPO} && python3 -m venv --upgrade .venv" >&2
  exit 1
fi
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

if ! "${PYTHON}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False on this node." >&2
  exit 1
fi

# Require PyG neighbor-sampling backend so train/eval do not fall back to full-graph encode (OOM on PrimeKG).
if ! "${PYTHON}" -c "
from kgml_new.training.hgt_train import neighbor_sampler_backend_available
import sys
sys.exit(0 if neighbor_sampler_backend_available() else 1)
" 2>/dev/null
then
  echo "ERROR: Neighbor sampling backend unavailable (need torch_sparse + torch_scatter in .venv; run scripts/bootstrap_gpu_venv.sh)." >&2
  echo "       Full-graph HGT would OOM on PrimeKG; refusing to run. See EXPERIMENTS.md HGT neighbor sampling." >&2
  exit 2
fi

TASK="${SLURM_ARRAY_TASK_ID:?}"
SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-64}"
EPOCHS="${EPOCHS:-100}"
EXTRA_HGT_ARGS=()

PRIMEKG_INPUT="${PRIMEKG_INPUT:-${REPO}/data/primekg.tsv}"
FB15K237_INPUT="${FB15K237_INPUT:-${REPO}/data/fb15k-237/fb15k-237.tsv}"

if [[ "${PRIMEKG_INPUT}" != /* ]]; then
  PRIMEKG_INPUT="${REPO}/${PRIMEKG_INPUT}"
fi
if [[ "${FB15K237_INPUT}" != /* ]]; then
  FB15K237_INPUT="${REPO}/${FB15K237_INPUT}"
fi

DATASETS=(primekg fb15k237)
SPLITS=(edge node)

DATASET_IDX=$((TASK / 2))
SPLIT_IDX=$((TASK % 2))
DATASET="${DATASETS[$DATASET_IDX]}"
SPLIT="${SPLITS[$SPLIT_IDX]}"

if [[ "${DATASET}" == "primekg" ]]; then
  INPUT="${PRIMEKG_INPUT}"
  REL="${HGT_RELATION_PRIMEKG:-indication}"
else
  INPUT="${FB15K237_INPUT}"
  REL="${HGT_RELATION_FB15K237:-}"
  if [[ -z "${REL}" ]]; then
    echo "ERROR: Set HGT_RELATION_FB15K237 to a relation name present in the FB15k TSV." >&2
    exit 1
  fi
fi

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input not found: ${INPUT}" >&2
  exit 1
fi

TYPE_EXTRA=()
if [[ -n "${HGT_SOURCE_NODE_TYPE:-}" ]]; then
  TYPE_EXTRA+=(--source-node-type "${HGT_SOURCE_NODE_TYPE}")
fi
if [[ -n "${HGT_TARGET_NODE_TYPE:-}" ]]; then
  TYPE_EXTRA+=(--target-node-type "${HGT_TARGET_NODE_TYPE}")
fi

if [[ "${DATASET}" == "primekg" ]]; then
  CSV_EXTRA=()
else
  CSV_EXTRA=(
    --source-col source
    --relation-col relation
    --target-col target
  )
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT_JSON="${REPO}/results/slurm/hgt-${DATASET}-${SPLIT}-s${SEED}-${JOB_TAG}.json"

echo "=== hgt_primekg_fb15k_split_array host=$(hostname) task=${TASK} dataset=${DATASET} split=${SPLIT} relation=${REL} ==="

set -x
"${PYTHON}" -m kgml_new.scripts.run_hgt \
  --input "${INPUT}" \
  --input-format auto \
  "${CSV_EXTRA[@]}" \
  --relation "${REL}" \
  "${TYPE_EXTRA[@]}" \
  --neighbor-sampling \
  --split-protocol "${SPLIT}" \
  --val-ratio "${VAL_RATIO:-0.1}" \
  --test-ratio "${TEST_RATIO:-0.1}" \
  --seed "${SEED}" \
  --in-dim "${IN_DIM}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --num-layers "${NUM_LAYERS:-2}" \
  --num-heads "${NUM_HEADS:-4}" \
  --dropout "${DROPOUT:-0.1}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --negative-sampling-mode "${NEGATIVE_SAMPLING_MODE:-global}" \
  "${EXTRA_HGT_ARGS[@]}" \
  --output "${OUT_JSON}"
set +x

echo "Wrote ${OUT_JSON}"
