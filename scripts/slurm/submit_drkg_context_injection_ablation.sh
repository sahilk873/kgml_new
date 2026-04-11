#!/usr/bin/env bash
# Submit full-DRKG context-injection ablations for BOTH splits (edge + node).
#
# This submits two array jobs:
#   1) SPLIT=edge
#   2) SPLIT=node
#
# Each array uses scripts/slurm/drkg_context_injection_ablation_array.slurm
# (8 tasks: 4 modes x 2 controls).
#
# Required relation caches:
#   RANDOM_REL_CACHE        (default: cache/drkg-relations-random.pt)
#   SHUFFLED_SEM_REL_CACHE  (default: cache/drkg-relations-sapbert-shuffled.pt)
#
# Optional (keep fixed defaults unless intentionally changed):
#   EPOCHS, SEED, IN_DIM, DECODER, NEGATIVES_PER_POS, DRKG_INPUT,
#   PREPARED_CACHE_EDGE, PREPARED_CACHE_NODE, PARTITION, ACCOUNT, QOS
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm results/slurm

if [[ ! -f "${REPO}/scripts/slurm/drkg_context_injection_ablation_array.slurm" ]]; then
  echo "ERROR: Missing array script scripts/slurm/drkg_context_injection_ablation_array.slurm" >&2
  exit 1
fi

INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"
RANDOM_REL_CACHE="${RANDOM_REL_CACHE:-${REPO}/cache/drkg-relations-random.pt}"
SHUFFLED_SEM_REL_CACHE="${SHUFFLED_SEM_REL_CACHE:-${REPO}/cache/drkg-relations-sapbert-shuffled.pt}"

if [[ "${INPUT}" != /* ]]; then
  INPUT="${REPO}/${INPUT}"
fi
if [[ "${RANDOM_REL_CACHE}" != /* ]]; then
  RANDOM_REL_CACHE="${REPO}/${RANDOM_REL_CACHE}"
fi
if [[ "${SHUFFLED_SEM_REL_CACHE}" != /* ]]; then
  SHUFFLED_SEM_REL_CACHE="${REPO}/${SHUFFLED_SEM_REL_CACHE}"
fi

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: DRKG_INPUT not found: ${INPUT}" >&2
  exit 1
fi
if [[ ! -f "${RANDOM_REL_CACHE}" ]]; then
  echo "ERROR: RANDOM_REL_CACHE not found: ${RANDOM_REL_CACHE}" >&2
  echo "  Example generation:" >&2
  echo "    python -m kgml_new.scripts.generate_relation_embeddings --input \"${INPUT}\" --output \"${RANDOM_REL_CACHE}\" --embedding-model random" >&2
  exit 1
fi
if [[ ! -f "${SHUFFLED_SEM_REL_CACHE}" ]]; then
  echo "ERROR: SHUFFLED_SEM_REL_CACHE not found: ${SHUFFLED_SEM_REL_CACHE}" >&2
  echo "  Provide a shuffled semantic cache path with:" >&2
  echo "    export SHUFFLED_SEM_REL_CACHE=/absolute/or/relative/path/to/cache.pt" >&2
  exit 1
fi

SEED="${SEED:-42}"
PREPARED_CACHE_EDGE="${PREPARED_CACHE_EDGE:-${REPO}/cache/drkg-prepared_edge_s${SEED}.pkl}"
PREPARED_CACHE_NODE="${PREPARED_CACHE_NODE:-${REPO}/cache/drkg-prepared_node_s${SEED}.pkl}"
if [[ "${PREPARED_CACHE_EDGE}" != /* ]]; then
  PREPARED_CACHE_EDGE="${REPO}/${PREPARED_CACHE_EDGE}"
fi
if [[ "${PREPARED_CACHE_NODE}" != /* ]]; then
  PREPARED_CACHE_NODE="${REPO}/${PREPARED_CACHE_NODE}"
fi
if [[ ! -f "${PREPARED_CACHE_EDGE}" ]]; then
  echo "ERROR: Edge prepared cache not found: ${PREPARED_CACHE_EDGE}" >&2
  exit 1
fi
if [[ ! -f "${PREPARED_CACHE_NODE}" ]]; then
  echo "ERROR: Node prepared cache not found: ${PREPARED_CACHE_NODE}" >&2
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

COMMON_EXPORTS="ALL,DRKG_INPUT=${INPUT},RANDOM_REL_CACHE=${RANDOM_REL_CACHE},SHUFFLED_SEM_REL_CACHE=${SHUFFLED_SEM_REL_CACHE},SEED=${SEED}"

EDGE_ID="$(
  sbatch "${SBATCH_ARGS[@]}" \
    --export="${COMMON_EXPORTS},SPLIT=edge,PREPARED_CACHE=${PREPARED_CACHE_EDGE}" \
    "${REPO}/scripts/slurm/drkg_context_injection_ablation_array.slurm"
)"
NODE_ID="$(
  sbatch "${SBATCH_ARGS[@]}" \
    --export="${COMMON_EXPORTS},SPLIT=node,PREPARED_CACHE=${PREPARED_CACHE_NODE}" \
    "${REPO}/scripts/slurm/drkg_context_injection_ablation_array.slurm"
)"

echo ""
echo "Submitted context-injection ablation arrays:"
echo "  edge split: ${EDGE_ID}"
echo "  node split: ${NODE_ID}"
echo ""
echo "Monitor queue:"
echo "  squeue -u \"\$USER\" | egrep '${EDGE_ID}|${NODE_ID}'"
echo ""
echo "Monitor until completion:"
echo "  LOG_PREFIX=drkg-ctxabl ./scripts/slurm/monitor_slurm_jobs.sh ${EDGE_ID} --poll 30"
echo "  LOG_PREFIX=drkg-ctxabl ./scripts/slurm/monitor_slurm_jobs.sh ${NODE_ID} --poll 30"
echo ""
echo "Expected outputs:"
echo "  results/slurm/drkg_ctxabl_edge_<mode>_<control>_${EDGE_ID}_<task>.json"
echo "  results/slurm/drkg_ctxabl_node_<mode>_<control>_${NODE_ID}_<task>.json"
