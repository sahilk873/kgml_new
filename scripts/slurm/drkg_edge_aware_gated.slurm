#!/bin/bash
# DRKG link prediction: edge-aware GraphSAGE with relation gating (gated) + SapBERT cache.
#
# Optional: NEIGHBOR_AGGR=mean|max (default mean).
#
#SBATCH --job-name=drkg-ea-gated
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/drkg-ea-gated-%j.out
#SBATCH --error=logs/slurm/drkg-ea-gated-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=/dev/null
source "${REPO}/scripts/slurm/drkg_gpu_common.sh"

NEIGHBOR_AGGR="${NEIGHBOR_AGGR:-mean}"

JOB_TAG="${SLURM_JOB_ID}"
LOG="${REPO}/logs/slurm/run-drkg-ea-gated-${JOB_TAG}.log"
OUT="${REPO}/results/slurm/drkg_edge_aware_gated_${NEIGHBOR_AGGR}_${JOB_TAG}.json"

echo "=== drkg_edge_aware_gated host=$(hostname) job=${JOB_TAG} aggr=${NEIGHBOR_AGGR} ==="
bash "${REPO}/scripts/diagnose_gpu_env.sh" || true

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA not available." >&2
  exit 1
fi

if [[ ! -f "${REL_CACHE}" ]]; then
  echo "WARNING: REL_CACHE not found at ${REL_CACHE}" >&2
  echo "  Build SapBERT relation embeddings once, e.g.:" >&2
  echo "    python -m kgml_new.scripts.generate_relation_embeddings --input \"\${INPUT}\" --output \"\${REL_CACHE}\" --embedding-model sapbert" >&2
fi

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method edge_aware_sage \
  --edge-relation-mode gated \
  --neighbor-aggr "${NEIGHBOR_AGGR}" \
  --semantic-cache "${REL_CACHE}" \
  --embedding-model sapbert \
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
