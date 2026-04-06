#!/bin/bash
# DRKG link prediction: node2vec with existing prepared cache.
#
# NODE2VEC_EPOCHS overrides EPOCHS for this job only (default: same as EPOCHS).
#
#SBATCH --job-name=drkg-n2v
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/drkg-n2v-%j.out
#SBATCH --error=logs/slurm/drkg-n2v-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=/dev/null
source "${REPO}/scripts/slurm/drkg_gpu_common.sh"

N2V_EPOCHS="${NODE2VEC_EPOCHS:-${EPOCHS}}"

JOB_TAG="${SLURM_JOB_ID}"
LOG="${REPO}/logs/slurm/run-drkg-n2v-${JOB_TAG}.log"
OUT="${REPO}/results/slurm/drkg_node2vec_${JOB_TAG}.json"

echo "=== drkg_node2vec host=$(hostname) job=${JOB_TAG} epochs=${N2V_EPOCHS} ==="
bash "${REPO}/scripts/diagnose_gpu_env.sh" || true

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA not available." >&2
  exit 1
fi

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method node2vec \
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
  --epochs "${N2V_EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}"
set +x

echo "Wrote ${OUT}"
