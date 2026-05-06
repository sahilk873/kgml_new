#!/bin/bash
# KG-completion sweep over PrimeKG and FB15k-237 using triple prediction (h, r, t).
#
# Array tasks 0-7:
#   0  primekg   baseline_sage     DistMult
#   1  primekg   edge_aware_sage   DistMult
#   2  primekg   baseline_gcn      DistMult
#   3  primekg   rgcn              DistMult
#   4  fb15k237  baseline_sage     DistMult
#   5  fb15k237  edge_aware_sage   DistMult
#   6  fb15k237  baseline_gcn      DistMult
#   7  fb15k237  rgcn              DistMult
#
# Uses random relation vectors unless REL_CACHE / semantic settings are provided.
#
#SBATCH --job-name=kgml-kgcomp-pk-fb15k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/slurm/kgml-kgcomp-pk-fb15k-%A_%a.out
#SBATCH --error=logs/slurm/kgml-kgcomp-pk-fb15k-%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"
mkdir -p logs/slurm results/slurm

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
EPOCHS="${EPOCHS:-100}"
DECODER="${DECODER:-distmult}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"

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
if [[ -z "${TASK}" ]] || [[ "${TASK}" -lt 0 ]] || [[ "${TASK}" -gt 7 ]]; then
  echo "ERROR: Invalid SLURM_ARRAY_TASK_ID=${TASK} (expected 0-7)" >&2
  exit 1
fi

DATASETS=(primekg fb15k237)
METHODS=(baseline_sage edge_aware_sage baseline_gcn rgcn)
DATASET_IDX=$((TASK / 4))
METHOD_IDX=$((TASK % 4))
DATASET="${DATASETS[$DATASET_IDX]}"
METHOD="${METHODS[$METHOD_IDX]}"

if [[ "${DATASET}" == "primekg" ]]; then
  INPUT="${PRIMEKG_INPUT}"
else
  INPUT="${FB15K237_INPUT}"
fi

COMMON_ARGS=(
  --require-cuda
  --task kg_completion
  --method "${METHOD}"
  --input "${INPUT}"
  --input-format auto
  --split-protocol "${SPLIT:-edge}"
  --seed "${SEED}"
  --in-dim "${IN_DIM}"
  --decoder "${DECODER}"
  --negatives-per-pos "${NEGATIVES_PER_POS}"
  --epochs "${EPOCHS}"
  --log-level INFO
)

# Keep relation features lightweight for the first pass: random relation vectors.
if [[ "${METHOD}" == "edge_aware_sage" ]]; then
  COMMON_ARGS+=(--no-semantic --embedding-model random)
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${TASK}"
OUT="${REPO}/results/slurm/${DATASET}_${METHOD}_${JOB_TAG}.json"
LOG="${REPO}/logs/slurm/run-${DATASET}-${METHOD}-${JOB_TAG}.log"

echo "=== primekg_fb15k_kg_completion_array host=$(hostname) task=${TASK} dataset=${DATASET} method=${METHOD} split=${SPLIT:-edge} ==="

set -x
python -m kgml_new.scripts.run_gpu_method \
  "${COMMON_ARGS[@]}" \
  --log-file "${LOG}" \
  --output "${OUT}"
set +x

echo "Wrote ${OUT}"
