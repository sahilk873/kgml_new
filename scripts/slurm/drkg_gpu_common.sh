# Shared setup for DRKG GPU jobs using an existing prepared link-prediction cache.
# Source from scripts under scripts/slurm/ after: REPO="${SLURM_SUBMIT_DIR:-$(pwd)}" && cd "$REPO"
#
# Expects: .venv at repo root, same env vars as drkg_full_experiments_array.slurm
# (DRKG_INPUT, PREPARED_CACHE, SPLIT, SEED, REL_CACHE, MAX_EDGES, etc.).

if [[ -z "${REPO:-}" ]]; then
  REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
cd "$REPO"
mkdir -p logs/slurm results/slurm cache

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in $REPO — create venv and pip install -e . first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

EPOCHS="${EPOCHS:-100}"
SPLIT="${SPLIT:-edge}"
SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-64}"
DECODER="${DECODER:-dot}"
NEGATIVES_PER_POS="${NEGATIVES_PER_POS:-20}"
USE_PREPARED_DATASET_CACHE="${USE_PREPARED_DATASET_CACHE:-1}"

INPUT="${DRKG_INPUT:-${REPO}/drkg.tsv}"
REL_CACHE="${REL_CACHE:-cache/drkg-relations-sapbert.pt}"

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
    PREPARED_CACHE="${REPO}/cache/drkg-prepared_${SPLIT}_s${SEED}.pkl"
  fi
  if [[ ! -f "${PREPARED_CACHE}" ]]; then
    echo "ERROR: Prepared dataset cache not found: ${PREPARED_CACHE}" >&2
    echo "  Build with: sbatch scripts/slurm/drkg_build_prepared_cache.slurm" >&2
    exit 1
  fi
  PREPARED_EXTRA=(--prepared-dataset-cache "${PREPARED_CACHE}")
  echo "Using prepared cache: ${PREPARED_CACHE}"
fi

NEG_EXTRA=()
if [[ -n "${NEGATIVE_SAMPLING_MODE:-}" ]]; then
  NEG_EXTRA=(--negative-sampling-mode "${NEGATIVE_SAMPLING_MODE}")
fi
