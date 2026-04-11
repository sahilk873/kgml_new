#!/usr/bin/env bash
# Submit CPU cache build, then GPU experiment array only after it succeeds (no idle GPUs).
#
# From repo root, set any env vars you need (same as the two .slurm files), then:
#   export DRKG_INPUT=cache/drkg.graph.pkl   # example
#   ./scripts/slurm/submit_drkg_full_with_cache.sh
#
# Or one line:
#   DRKG_INPUT=cache/drkg.graph.pkl ./scripts/slurm/submit_drkg_full_with_cache.sh
#
# Node-disjoint split (builds cache/drkg-prepared_node_s<SEED>.pkl, not the edge cache):
#   SPLIT=node ./scripts/slurm/submit_drkg_full_with_cache.sh
#
# Uses sbatch --dependency=afterok:BUILD so if the cache job fails, the GPU array never starts.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm

SPLIT="${SPLIT:-edge}"
echo "Using SPLIT=${SPLIT} (default prepared cache: cache/drkg-prepared_${SPLIT}_s<SEED>.pkl)"

# CPU partition name is site-specific (hpc6: batch, smp, amd — not "cpu").
BUILD_PARTITION="${BUILD_PARTITION:-batch}"
BUILD_ID="$(sbatch --parsable --partition="${BUILD_PARTITION}" --export=ALL,SPLIT="${SPLIT}" "${REPO}/scripts/slurm/drkg_build_prepared_cache.slurm")"
echo "Prepared-cache build job: ${BUILD_ID}"

GPU_ID="$(sbatch --parsable --dependency=afterok:"${BUILD_ID}" --export=ALL,SPLIT="${SPLIT}" "${REPO}/scripts/slurm/drkg_full_experiments_array.slurm")"
echo "GPU experiment array (starts after cache job succeeds): ${GPU_ID}"

echo "Monitor: ./scripts/slurm/monitor_slurm_jobs.sh ${GPU_ID}"
