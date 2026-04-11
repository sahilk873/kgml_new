#!/usr/bin/env bash
# Submit node-split semantic-vs-random sweep (36 GPU tasks) without wasting GPUs on export.
#
# Default: submits a CPU job that builds both prepared caches (d128 + d256 for SPLIT=node),
# then submits the GPU array with --dependency=afterok:<cpu_job> so training starts only when
# caches exist.
#
# If caches are already built:
#   SKIP_PREPARED_CACHE_BUILD=1 ./scripts/slurm/submit_drkg_node_semantic_random_dim_sweep.sh
#
# Optional overrides (export before running):
#   DRKG_INPUT=cache/drkg.graph.pkl
#   SPLIT=node SEED=42
#   BUILD_PARTITION=batch   # site-specific CPU queue
#   NODE_PREP_DIMS="128 256"
#   SAPBERT_REL_CACHE=cache/drkg-relations-sapbert.pt
#   RANDOM_REL_CACHE=cache/drkg-relations-random.pt
#   EPOCHS=100
#
# Then:
#   ./scripts/slurm/submit_drkg_node_semantic_random_dim_sweep.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm results/slurm cache

BUILD_PARTITION="${BUILD_PARTITION:-batch}"
SPLIT="${SPLIT:-node}"

if [[ "${SKIP_PREPARED_CACHE_BUILD:-0}" == "1" ]]; then
  GPU_ID="$(sbatch --parsable --export=ALL,SPLIT="${SPLIT}" "${REPO}/scripts/slurm/drkg_node_semantic_random_dim_sweep_array.slurm")"
  echo ""
  echo "Submitted GPU sweep only (SKIP_PREPARED_CACHE_BUILD=1):"
  echo "  GPU job: ${GPU_ID}"
else
  BUILD_ID="$(sbatch --parsable --partition="${BUILD_PARTITION}" --export=ALL,SPLIT="${SPLIT}" "${REPO}/scripts/slurm/drkg_build_prepared_node_dim_caches.slurm")"
  echo "CPU prepared-cache build (d128 + d256 for node split): ${BUILD_ID}"

  GPU_ID="$(sbatch --parsable --dependency=afterok:"${BUILD_ID}" --export=ALL,SPLIT="${SPLIT}" "${REPO}/scripts/slurm/drkg_node_semantic_random_dim_sweep_array.slurm")"
  echo "GPU sweep (starts after CPU build succeeds): ${GPU_ID}"
fi

echo ""
echo "Tasks: 36 (array 0-35)"
echo ""
echo "Queue:"
echo "  squeue -u \"\$USER\" | rg \"${GPU_ID}\""
echo ""
echo "Monitor GPU sweep:"
echo "  LOG_PREFIX=drkg-node-semrand ${REPO}/scripts/slurm/monitor_slurm_jobs.sh ${GPU_ID}"
echo ""
echo "Early stderr (GPU):"
echo "  tail -f logs/slurm/drkg-node-semrand-${GPU_ID}_0.err"
