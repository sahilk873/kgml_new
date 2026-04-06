#!/bin/bash
# DRKG link prediction: edge-aware GraphSAGE with basis mixture + SapBERT cache.
#
# Optional env:
#   NEIGHBOR_AGGR=mean|max (default mean)
#   NUM_RELATION_BASES (default 4)
#   SEMANTIC_ALIGNMENT_LAMBDA — if >0, enables semantic alignment regularizer (basis_mixture)
#
#SBATCH --job-name=drkg-ea-basis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/drkg-ea-basis-%j.out
#SBATCH --error=logs/slurm/drkg-ea-basis-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=/dev/null
source "${REPO}/scripts/slurm/drkg_gpu_common.sh"

NEIGHBOR_AGGR="${NEIGHBOR_AGGR:-mean}"
NUM_RELATION_BASES="${NUM_RELATION_BASES:-4}"
SEMANTIC_ALIGNMENT_LAMBDA="${SEMANTIC_ALIGNMENT_LAMBDA:-0.0}"

BASIS_EXTRA=(
  --num-relation-bases "${NUM_RELATION_BASES}"
  --semantic-alignment-lambda "${SEMANTIC_ALIGNMENT_LAMBDA}"
)

JOB_TAG="${SLURM_JOB_ID}"
LOG="${REPO}/logs/slurm/run-drkg-ea-basis-${JOB_TAG}.log"
OUT="${REPO}/results/slurm/drkg_edge_aware_basis_mixture_b${NUM_RELATION_BASES}_${JOB_TAG}.json"

echo "=== drkg_edge_aware_basis_mixture host=$(hostname) job=${JOB_TAG} aggr=${NEIGHBOR_AGGR} bases=${NUM_RELATION_BASES} align_lambda=${SEMANTIC_ALIGNMENT_LAMBDA} ==="
bash "${REPO}/scripts/diagnose_gpu_env.sh" || true

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA not available." >&2
  exit 1
fi

if [[ ! -f "${REL_CACHE}" ]]; then
  echo "WARNING: REL_CACHE not found at ${REL_CACHE}" >&2
fi

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method edge_aware_sage \
  --edge-relation-mode basis_mixture \
  --neighbor-aggr "${NEIGHBOR_AGGR}" \
  --semantic-cache "${REL_CACHE}" \
  --embedding-model sapbert \
  "${BASIS_EXTRA[@]}" \
  --input "${INPUT}" \
  --input-format auto \
  --split-protocol "${SPLIT}" \
  --seed "${SEED}" \
  --in-dim "${IN_DIM}" \
  --decoder "${DECODER}" \
  --negatives-per-pos "${NEGATIVES_PER_POS}" \
  "${SHUFFLE_EXTRA[@]}" \
  "${NEG_EXTRA[@]}" \
  "${MAX_EXTRA[@]}" \
  "${DIV_EXTRA[@]}" \
  "${PREPARED_EXTRA[@]}" \
  --epochs "${EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}"
set +x

echo "Wrote ${OUT}"
