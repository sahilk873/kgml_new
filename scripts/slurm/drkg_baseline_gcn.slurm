#!/bin/bash
# DRKG link prediction: baseline GCN with existing prepared cache (no cache build on GPU).
#
# Same env as drkg_full_experiments_array.slurm (DRKG_INPUT, PREPARED_CACHE, SPLIT, SEED, …).
#
#SBATCH --job-name=drkg-gcn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/drkg-gcn-%j.out
#SBATCH --error=logs/slurm/drkg-gcn-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck source=/dev/null
source "${REPO}/scripts/slurm/drkg_gpu_common.sh"

JOB_TAG="${SLURM_JOB_ID}"
LOG="${REPO}/logs/slurm/run-drkg-gcn-${JOB_TAG}.log"
OUT="${REPO}/results/slurm/drkg_baseline_gcn_${JOB_TAG}.json"

echo "=== drkg_baseline_gcn host=$(hostname) job=${JOB_TAG} ==="
bash "${REPO}/scripts/diagnose_gpu_env.sh" || true

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: CUDA not available." >&2
  exit 1
fi

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method baseline_gcn \
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
