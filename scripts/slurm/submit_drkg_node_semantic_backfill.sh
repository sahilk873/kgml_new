#!/usr/bin/env bash
# Submit missing node-split semantic backfill runs (4-task array).
#
# Runs:
#   film pool semantic
#   concat semantic
#   gated semantic
#   basis_mixture semantic
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm results/slurm

INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"
SEED="${SEED:-42}"
PREPARED_CACHE="${PREPARED_CACHE:-${REPO}/cache/drkg-prepared_node_s${SEED}.pkl}"
REL_CACHE="${REL_CACHE:-${REPO}/cache/drkg-relations-sapbert.pt}"

if [[ "${INPUT}" != /* ]]; then
  INPUT="${REPO}/${INPUT}"
fi
if [[ "${PREPARED_CACHE}" != /* ]]; then
  PREPARED_CACHE="${REPO}/${PREPARED_CACHE}"
fi
if [[ "${REL_CACHE}" != /* ]]; then
  REL_CACHE="${REPO}/${REL_CACHE}"
fi

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: DRKG_INPUT not found: ${INPUT}" >&2
  exit 1
fi
if [[ ! -f "${PREPARED_CACHE}" ]]; then
  echo "ERROR: Node prepared cache not found: ${PREPARED_CACHE}" >&2
  exit 1
fi
if [[ ! -f "${REL_CACHE}" ]]; then
  echo "ERROR: Semantic relation cache not found: ${REL_CACHE}" >&2
  exit 1
fi

SBATCH_ARGS=("--parsable")
if [[ -n "${PARTITION:-}" ]]; then
  SBATCH_ARGS+=("--partition=${PARTITION}")
fi
if [[ -n "${ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=("--account=${ACCOUNT}")
fi
if [[ -n "${QOS:-}" ]]; then
  SBATCH_ARGS+=("--qos=${QOS}")
fi

JOB_ID="$(
  sbatch "${SBATCH_ARGS[@]}" \
    --export="ALL,DRKG_INPUT=${INPUT},PREPARED_CACHE=${PREPARED_CACHE},REL_CACHE=${REL_CACHE},SEED=${SEED}" \
    "${REPO}/scripts/slurm/drkg_node_semantic_backfill_array.slurm"
)"

echo ""
echo "Submitted node semantic backfill array: ${JOB_ID}"
echo ""
echo "Queue:"
echo "  squeue -u \"\$USER\" | egrep '${JOB_ID}|drkg-node-sem-backfill'"
echo ""
echo "Monitor:"
echo "  LOG_PREFIX=drkg-node-sem-backfill ./scripts/slurm/monitor_slurm_jobs.sh ${JOB_ID} --poll 30"
echo ""
echo "Expected outputs:"
echo "  results/slurm/drkg_node_semantic_<mode>_pool_${JOB_ID}_<task>.json"
