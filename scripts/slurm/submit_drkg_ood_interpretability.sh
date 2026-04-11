#!/usr/bin/env bash
# Submit CPU prepared-cache build, then GPU drkg_ood_interpretability_array (afterok dependency).
#
# Defaults: SPLIT=node (entity-inductive), OOD_SAVE_EDGE_CSV=0.
# Propagate any exported vars (DRKG_INPUT, HELD_OUT_RELATIONS, SEED, MAX_EDGES, …) via --export=ALL.
#
# Examples:
#   ./scripts/slurm/submit_drkg_ood_interpretability.sh
#   SPLIT=node HELD_OUT_RELATIONS='SomeRel' ./scripts/slurm/submit_drkg_ood_interpretability.sh
#
# After the array finishes, aggregate OOD JSON (adjust glob to your job id / paths):
#   python -m kgml_new.scripts.aggregate_ood_results \
#     --inputs results/slurm/drkg_ood_*.json \
#     --tag drkg_ood_node \
#     --baseline-embedding random \
#     --target-embedding sapbert \
#     --out-dir results/ood
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm results/ood

SPLIT="${SPLIT:-node}"
OOD_SAVE_EDGE_CSV="${OOD_SAVE_EDGE_CSV:-0}"
export SPLIT OOD_SAVE_EDGE_CSV

echo "Using SPLIT=${SPLIT} HELD_OUT_RELATIONS=${HELD_OUT_RELATIONS:-}(empty ok) OOD_SAVE_EDGE_CSV=${OOD_SAVE_EDGE_CSV}"

BUILD_PARTITION="${BUILD_PARTITION:-batch}"
BUILD_ID="$(sbatch --parsable --partition="${BUILD_PARTITION}" --export=ALL "${REPO}/scripts/slurm/drkg_build_prepared_cache.slurm")"
echo "Prepared-cache build job: ${BUILD_ID}"

GPU_ID="$(sbatch --parsable --dependency=afterok:${BUILD_ID} --export=ALL "${REPO}/scripts/slurm/drkg_ood_interpretability_array.slurm")"
echo "GPU OOD interpretability array: ${GPU_ID}"

echo "Monitor: ./scripts/slurm/monitor_slurm_jobs.sh ${GPU_ID}"
echo "Aggregate (after success), e.g.:"
echo "  python -m kgml_new.scripts.aggregate_ood_results --inputs results/slurm/drkg_ood_*.json --tag drkg_ood --baseline-embedding random --target-embedding sapbert --out-dir results/ood"
