#!/usr/bin/env bash
# Submit DRKG GPU jobs that use the existing prepared cache (baseline GCN, node2vec,
# edge-aware gated, edge-aware basis mixture). Does not rebuild the cache.
#
# Export the same variables as for drkg_full_experiments_array.slurm, e.g.:
#   export DRKG_INPUT=cache/drkg.graph.pkl
#   export PREPARED_CACHE=cache/drkg-prepared_edge_s42.pkl   # optional if defaults match SPLIT/SEED
#
# Then:
#   ./scripts/slurm/submit_drkg_prepared_extra_methods.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm results/slurm

submit_one() {
  local script="$1"
  local id
  id="$(sbatch --parsable "${REPO}/scripts/slurm/${script}")"
  printf '%s\n' "$id"
}

GCN_ID="$(submit_one drkg_baseline_gcn.slurm)"
N2V_ID="$(submit_one drkg_node2vec.slurm)"
GATED_ID="$(submit_one drkg_edge_aware_gated.slurm)"
BASIS_ID="$(submit_one drkg_edge_aware_basis_mixture.slurm)"

echo ""
echo "Submitted (job IDs):"
echo "  drkg_baseline_gcn.slurm              -> ${GCN_ID}"
echo "  drkg_node2vec.slurm                  -> ${N2V_ID}"
echo "  drkg_edge_aware_gated.slurm          -> ${GATED_ID}"
echo "  drkg_edge_aware_basis_mixture.slurm  -> ${BASIS_ID}"
echo ""
echo "Queue:"
echo "  squeue -u \"\$USER\" | egrep '${GCN_ID}|${N2V_ID}|${GATED_ID}|${BASIS_ID}'"
echo ""
echo "Monitor until a job finishes (set LOG_PREFIX to match Slurm output name stem):"
echo "  LOG_PREFIX=drkg-gcn    ${REPO}/scripts/slurm/monitor_slurm_jobs.sh ${GCN_ID}"
echo "  LOG_PREFIX=drkg-n2v    ${REPO}/scripts/slurm/monitor_slurm_jobs.sh ${N2V_ID}"
echo "  LOG_PREFIX=drkg-ea-gated  ${REPO}/scripts/slurm/monitor_slurm_jobs.sh ${GATED_ID}"
echo "  LOG_PREFIX=drkg-ea-basis  ${REPO}/scripts/slurm/monitor_slurm_jobs.sh ${BASIS_ID}"
echo ""
echo "Early stderr (replace JOBID):"
echo "  tail -f logs/slurm/drkg-gcn-JOBID.err"
echo "  tail -f logs/slurm/drkg-n2v-JOBID.err"
echo "  tail -f logs/slurm/drkg-ea-gated-JOBID.err"
echo "  tail -f logs/slurm/drkg-ea-basis-JOBID.err"
