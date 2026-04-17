#!/usr/bin/env bash
# Submit CPU cache build first, then GPU sweep after successful cache completion.
# This prevents GPUs from waiting on cache generation.
#
# Usage:
#   ./scripts/slurm/submit_primekg_fb15k_full_with_cache.sh
#
# Optional overrides:
#   BUILD_PARTITION=batch
#   SKIP_PREPARED_CACHE_BUILD=1
#   PRIMEKG_INPUT=data/kg.csv
#   FB15K237_INPUT=data/fb15k-237/fb15k-237.tsv
#   SEED=42 IN_DIM=128 EDGE_DIM=128 EPOCHS=100

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
mkdir -p logs/slurm results/slurm cache

BUILD_PARTITION="${BUILD_PARTITION:-batch}"

if [[ "${SKIP_PREPARED_CACHE_BUILD:-0}" == "1" ]]; then
  GPU_ID="$(sbatch --parsable --export=ALL "${REPO}/scripts/slurm/primekg_fb15k_full_sweep_array.slurm")"
  echo "Submitted GPU sweep only (SKIP_PREPARED_CACHE_BUILD=1): ${GPU_ID}"
else
  BUILD_ID="$(sbatch --parsable --partition="${BUILD_PARTITION}" --export=ALL "${REPO}/scripts/slurm/primekg_fb15k_build_prepared_caches.slurm")"
  echo "Submitted CPU cache build: ${BUILD_ID}"
  GPU_ID="$(sbatch --parsable --dependency=afterok:"${BUILD_ID}" --export=ALL "${REPO}/scripts/slurm/primekg_fb15k_full_sweep_array.slurm")"
  echo "Submitted GPU sweep (afterok:${BUILD_ID}): ${GPU_ID}"
fi

echo ""
echo "Queue:"
echo "  squeue -u \"\$USER\" | rg \"${GPU_ID}\""
echo ""
echo "Monitor:"
echo "  LOG_PREFIX=kgml-primekg-fb15k-sweep ${REPO}/scripts/slurm/monitor_slurm_jobs.sh ${GPU_ID}"
echo ""
echo "Expected GPU tasks: 24 (array 0-23)"
