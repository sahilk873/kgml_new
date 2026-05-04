#!/bin/bash
# Train PrimeKG node2vec from data/kg.csv and save embeddings for UMAP.
#
# Submit from repo root:
#   sbatch scripts/slurm/run_node2vec_primekg_kgcsv_umap.slurm
#
# The UMAP-ready tensor will be:
#   results/slurm/artifacts/primekg_edge_node2vec_kgcsv_${SLURM_JOB_ID}.train_embeddings.pt

#SBATCH --job-name=kgml-node2vec-primekg-kgcsv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/scratch/pioneer/users/sxk2517/kgml_new}"
cd "${REPO}"
mkdir -p logs/slurm results/slurm/artifacts

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in ${REPO}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

INPUT="${PRIMEKG_KG_CSV:-${REPO}/data/kg.csv}"
if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: PrimeKG kg.csv not found: ${INPUT}" >&2
  exit 1
fi

if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False on this node." >&2
  exit 1
fi

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
EMBED_DIM="${EMBED_DIM:-64}"
BATCH_SIZE="${BATCH_SIZE:-512}"
WALK_LENGTH="${WALK_LENGTH:-20}"
CONTEXT_SIZE="${CONTEXT_SIZE:-15}"
WALKS_PER_NODE="${WALKS_PER_NODE:-4}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
SPLIT="${SPLIT:-edge}"

JOB_TAG="${SLURM_JOB_ID:-local}"
OUT="${REPO}/results/slurm/primekg_${SPLIT}_node2vec_kgcsv_${JOB_TAG}.json"
LOG="${REPO}/logs/slurm/run-primekg-${SPLIT}-node2vec-kgcsv-${JOB_TAG}.log"

echo "=== PrimeKG kg.csv node2vec host=$(hostname) split=${SPLIT} seed=${SEED} embed_dim=${EMBED_DIM} ==="
echo "Input: ${INPUT}"
echo "Output JSON: ${OUT}"

EMB="${REPO}/results/slurm/artifacts/$(basename "${OUT}" .json).train_embeddings.pt"
MODEL="${REPO}/results/slurm/artifacts/$(basename "${OUT}" .json).model.pt"

set -x
python scripts/train_node2vec_kgcsv_cache.py \
  --input "${INPUT}" \
  --output "${EMB}" \
  --model-output "${MODEL}" \
  --json-output "${OUT}" \
  --seed "${SEED}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --embed-dim "${EMBED_DIM}" \
  --walk-length "${WALK_LENGTH}" \
  --context-size "${CONTEXT_SIZE}" \
  --walks-per-node "${WALKS_PER_NODE}" 2>&1 | tee "${LOG}"
set +x

echo "Wrote ${OUT}"
echo "Wrote embeddings: ${EMB}"
echo "Wrote model: ${MODEL}"
