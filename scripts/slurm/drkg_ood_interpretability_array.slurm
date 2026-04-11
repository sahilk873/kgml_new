#!/bin/bash
# DRKG interpretability / harder-OOD bundle: node split (default), post-hoc OOD difficulty,
# optional relation holdout, and SapBERT vs random relation embeddings on edge-aware FiLM.
#
# Array tasks (6):
#   0 — baseline_sage mean
#   1 — baseline_sage max (pool)
#   2 — edge_aware_sage FiLM mean + SapBERT (semantic)
#   3 — edge_aware_sage FiLM max + SapBERT (semantic)
#   4 — edge_aware_sage FiLM mean + random relation embeddings (control)
#   5 — edge_aware_sage FiLM max + random relation embeddings (control)
#
# Every run passes --compute-ood-difficulty. Use aggregate_ood_results.py with
#   --baseline-embedding random --target-embedding sapbert
# on the result JSONs to summarize semantic-vs-random gaps per OOD bucket (paired by seed).
#
# Submit from repo root:
#   sbatch scripts/slurm/drkg_ood_interpretability_array.slurm
#
# Environment (export before sbatch or sbatch --export=ALL):
#   SPLIT — default node (entity-inductive; must match prepared cache).
#   DRKG_INPUT, REL_CACHE — same as drkg_full_experiments_array.slurm.
#   REL_RANDOM_CACHE — default cache/drkg-relations-random.pt (for tasks 4–5 --semantic-cache).
#   USE_PREPARED_DATASET_CACHE, PREPARED_CACHE, SEED, EPOCHS, IN_DIM, DECODER, NEGATIVES_PER_POS,
#   MAX_EDGES, NEGATIVE_SAMPLING_MODE, SHUFFLE_RELATIONS, DIVERSITY_BUCKETS_CACHE, NO_AUTO_DIVERSITY_CACHE,
#   HELD_OUT_RELATIONS — space-separated relation *names* held out from training (zero-shot at val/test).
#     When set, default PREPARED_CACHE becomes cache/drkg-prepared_${SPLIT}_s${SEED}_ho_<slug>.pkl
#     (build with the same names via export_prepared_link_prediction or drkg_build_prepared_cache.slurm).
#   OOD_OUTPUT_DIR — default ${REPO}/results/ood (per-edge CSV + OOD run naming).
#   OOD_SAVE_EDGE_CSV — set to 1 for --save-edge-predictions (disk-heavy on full DRKG; OK with MAX_EDGES).
#
# Build prepared cache for node split + holdout (example):
#   HELD_OUT_RELATIONS='SomeRelation' SPLIT=node sbatch scripts/slurm/drkg_build_prepared_cache.slurm
# Or manually:
#   python -m kgml_new.scripts.export_prepared_link_prediction \
#     --input cache/drkg.graph.pkl --split-protocol node --seed 42 \
#     --held-out-relations SomeRelation \
#     --output cache/drkg-prepared_node_s42_ho_SomeRelation.pkl
#
# If SHUFFLE_RELATIONS=1, set PREPARED_CACHE explicitly to a path built with --shuffle-relations
# (metadata must match).
#
#SBATCH --job-name=drkg-ood
#SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT
##SBATCH --qos=YOUR_QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-5%6
#SBATCH --output=logs/slurm/drkg-ood-%A_%a.out
#SBATCH --error=logs/slurm/drkg-ood-%A_%a.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
mkdir -p logs/slurm results/slurm results/ood cache

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in $REPO — create venv and pip install -e . first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

EPOCHS="${EPOCHS:-100}"
SPLIT="${SPLIT:-node}"
SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-64}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
USE_PREPARED_DATASET_CACHE="${USE_PREPARED_DATASET_CACHE:-1}"
INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"
REL_CACHE="${REL_CACHE:-cache/drkg-relations-sapbert.pt}"
REL_RANDOM_CACHE="${REL_RANDOM_CACHE:-cache/drkg-relations-random.pt}"
OOD_OUTPUT_DIR="${OOD_OUTPUT_DIR:-${REPO}/results/ood}"

if [[ "${INPUT}" != /* ]]; then
  INPUT="${REPO}/${INPUT}"
fi
if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: Input KG file not found: ${INPUT}" >&2
  exit 1
fi

if [[ "${REL_CACHE}" != /* ]]; then
  REL_CACHE="${REPO}/${REL_CACHE}"
fi
if [[ "${REL_RANDOM_CACHE}" != /* ]]; then
  REL_RANDOM_CACHE="${REPO}/${REL_RANDOM_CACHE}"
fi
if [[ "${OOD_OUTPUT_DIR}" != /* ]]; then
  OOD_OUTPUT_DIR="${REPO}/${OOD_OUTPUT_DIR}"
fi

DIV_EXTRA=()
DIV_CAND=""
if [[ -n "${DIVERSITY_BUCKETS_CACHE:-}" ]]; then
  DIV_CAND="${DIVERSITY_BUCKETS_CACHE}"
elif [[ -z "${NO_AUTO_DIVERSITY_CACHE:-}" ]] && [[ "${INPUT}" == *.graph.pkl ]]; then
  DIV_CAND="${INPUT%.graph.pkl}.diversity.pkl"
fi
if [[ -n "${DIV_CAND}" ]]; then
  if [[ "${DIV_CAND}" != /* ]]; then
    DIV_CAND="${REPO}/${DIV_CAND}"
  fi
  if [[ -f "${DIV_CAND}" ]]; then
    DIV_EXTRA=(--diversity-buckets-cache "${DIV_CAND}")
    echo "Using diversity buckets cache: ${DIV_CAND}"
  elif [[ -n "${DIVERSITY_BUCKETS_CACHE:-}" ]]; then
    echo "ERROR: DIVERSITY_BUCKETS_CACHE not found: ${DIV_CAND}" >&2
    exit 1
  fi
fi

MAX_EXTRA=()
if [[ -n "${MAX_EDGES:-}" ]]; then
  MAX_EXTRA=(--max-edges "${MAX_EDGES}")
fi

HOLDOUT_EXTRA=()
if [[ -n "${HELD_OUT_RELATIONS:-}" ]]; then
  read -r -a _hor <<< "${HELD_OUT_RELATIONS}"
  if [[ ${#_hor[@]} -gt 0 ]]; then
    HOLDOUT_EXTRA=(--held-out-relations "${_hor[@]}")
  fi
fi

SHUFFLE_EXTRA=()
if [[ "${SHUFFLE_RELATIONS:-0}" == "1" ]]; then
  SHUFFLE_EXTRA=(--shuffle-relations)
fi

PREPARED_EXTRA=()
if [[ "${USE_PREPARED_DATASET_CACHE}" == "1" ]]; then
  if [[ -n "${PREPARED_CACHE:-}" ]]; then
    if [[ "${PREPARED_CACHE}" != /* ]]; then
      PREPARED_CACHE="${REPO}/${PREPARED_CACHE}"
    fi
  else
    HO_PART=""
    if [[ -n "${HELD_OUT_RELATIONS:-}" ]]; then
      HO_SLUG=$(printf '%s' "${HELD_OUT_RELATIONS}" | tr -c 'a-zA-Z0-9_' '_' | tr -s '_' | sed 's/^_\|_$//g')
      if [[ -n "${HO_SLUG}" ]]; then
        HO_PART="_ho_${HO_SLUG}"
      fi
    fi
    PREPARED_CACHE="${REPO}/cache/drkg-prepared_${SPLIT}_s${SEED}${HO_PART}.pkl"
  fi
  if [[ ! -f "${PREPARED_CACHE}" ]]; then
    echo "ERROR: Prepared dataset cache not found: ${PREPARED_CACHE}" >&2
    echo "  Build: sbatch scripts/slurm/drkg_build_prepared_cache.slurm (same SPLIT, SEED, HELD_OUT_RELATIONS, …)" >&2
    echo "  Or: USE_PREPARED_DATASET_CACHE=0 (slow GPU prep)." >&2
    exit 1
  fi
  PREPARED_EXTRA=(--prepared-dataset-cache "${PREPARED_CACHE}")
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
LOG="${REPO}/logs/slurm/run-drkg-ood-${JOB_TAG}.log"

echo "=== drkg_ood_interpretability task=${SLURM_ARRAY_TASK_ID:-?} host=$(hostname) ==="
bash scripts/diagnose_gpu_env.sh || true

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "ERROR: torch.cuda.is_available() is False on this GPU node." >&2
  exit 1
fi

echo "Using prepared cache: ${PREPARED_CACHE:-(disabled)}"
echo "OOD output dir: ${OOD_OUTPUT_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    NAME=bsage_mean
    EXTRA=(--method baseline_sage --neighbor-aggr mean)
    ;;
  1)
    NAME=bsage_pool
    EXTRA=(--method baseline_sage --neighbor-aggr max)
    ;;
  2)
    NAME=easage_film_mean_semantic
    EXTRA=(
      --method edge_aware_sage
      --edge-relation-mode film
      --neighbor-aggr mean
      --semantic-cache "${REL_CACHE}"
      --embedding-model sapbert
    )
    ;;
  3)
    NAME=easage_film_pool_semantic
    EXTRA=(
      --method edge_aware_sage
      --edge-relation-mode film
      --neighbor-aggr max
      --semantic-cache "${REL_CACHE}"
      --embedding-model sapbert
    )
    ;;
  4)
    NAME=easage_film_mean_random
    EXTRA=(
      --method edge_aware_sage
      --edge-relation-mode film
      --neighbor-aggr mean
      --semantic-cache "${REL_RANDOM_CACHE}"
      --no-semantic
      --embedding-model random
    )
    ;;
  5)
    NAME=easage_film_pool_random
    EXTRA=(
      --method edge_aware_sage
      --edge-relation-mode film
      --neighbor-aggr max
      --semantic-cache "${REL_RANDOM_CACHE}"
      --no-semantic
      --embedding-model random
    )
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}" >&2
    exit 1
    ;;
esac

OUT="${REPO}/results/slurm/drkg_ood_${NAME}_${JOB_TAG}.json"
OOD_STEM="$(basename "${OUT}" .json)"
OOD_EXTRA=(
  --compute-ood-difficulty
  --ood-output-dir "${OOD_OUTPUT_DIR}"
  --ood-run-name "${OOD_STEM}"
)
if [[ "${OOD_SAVE_EDGE_CSV:-0}" == "1" ]]; then
  OOD_EXTRA+=(--save-edge-predictions)
fi

NEG_EXTRA=()
if [[ -n "${NEGATIVE_SAMPLING_MODE:-}" ]]; then
  NEG_EXTRA=(--negative-sampling-mode "${NEGATIVE_SAMPLING_MODE}")
fi

if [[ ! -f "${REL_CACHE}" ]] && [[ "${NAME}" == easage_*_semantic ]]; then
  echo "WARNING: REL_CACHE not found at ${REL_CACHE}" >&2
  echo "  Build: python -m kgml_new.scripts.generate_relation_embeddings --input \"\${INPUT}\" --output \"\${REL_CACHE}\" --embedding-model sapbert" >&2
fi
if [[ ! -f "${REL_RANDOM_CACHE}" ]] && [[ "${NAME}" == easage_*_random ]]; then
  echo "WARNING: REL_RANDOM_CACHE not found at ${REL_RANDOM_CACHE}" >&2
  echo "  Build: python -m kgml_new.scripts.generate_relation_embeddings --input \"\${INPUT}\" --output \"\${REL_RANDOM_CACHE}\" --embedding-model random" >&2
fi

set -x
python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --input "${INPUT}" \
  --input-format auto \
  --split-protocol "${SPLIT}" \
  --seed "${SEED}" \
  --in-dim "${IN_DIM}" \
  --decoder "${DECODER}" \
  --negatives-per-pos "${NEGATIVES_PER_POS}" \
  "${SHUFFLE_EXTRA[@]}" \
  "${HOLDOUT_EXTRA[@]}" \
  "${NEG_EXTRA[@]}" \
  "${MAX_EXTRA[@]}" \
  "${DIV_EXTRA[@]}" \
  "${PREPARED_EXTRA[@]}" \
  "${OOD_EXTRA[@]}" \
  --epochs "${EPOCHS}" \
  --log-file "${LOG}" \
  --log-level INFO \
  --output "${OUT}" \
  "${EXTRA[@]}"
set +x

echo "Wrote ${OUT}"
