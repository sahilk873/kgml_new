#!/bin/bash
# Build UMAP figures for PrimeKG + FB15k-237 (edge split, seed 42) from saved artifacts.
#
# Uses one GPU for HGT embedding export (encode); UMAP itself runs on CPU.
#
# Prerequisites:
#   - .venv at repo root with kgml-new installed
#   - Prepared caches (same as full sweep): cache/{dataset}-prepared_edge_s42_d128.pkl
#   - Artifact files referenced in figures/umap/manifest_*.json (update stems if your job IDs differ)
#
# Submit from repo root:
#   sbatch scripts/slurm/umap_primekg_fb15k_gpu.slurm
#
#SBATCH --job-name=kgml-umap-pk-fb15k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/kgml-umap-primekg-fb15k-%j.out
#SBATCH --error=logs/slurm/kgml-umap-primekg-fb15k-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO}"
mkdir -p logs/slurm figures/umap/embeddings figures/umap/out

if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found in ${REPO}" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

pip install -q -e ".[viz]"

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "WARNING: CUDA not visible; HGT export will fall back to CPU." >&2
fi

SEED="${SEED:-42}"
IN_DIM="${IN_DIM:-128}"
PKL_PRIME="${PKL_PRIME:-${REPO}/cache/primekg-prepared_edge_s${SEED}_d${IN_DIM}.pkl}"
PKL_FB="${PKL_FB:-${REPO}/cache/fb15k237-prepared_edge_s${SEED}_d${IN_DIM}.pkl}"

for p in "${PKL_PRIME}" "${PKL_FB}"; do
  if [[ ! -f "${p}" ]]; then
    echo "ERROR: Prepared cache missing: ${p}" >&2
    echo "Build with export_prepared_link_prediction (see EXPERIMENTS.md / primekg_fb15k_full_sweep_array.slurm)." >&2
    exit 1
  fi
done

PRIMEKG_INPUT="${PRIMEKG_INPUT:-${REPO}/data/primekg.tsv}"
FB15K_INPUT="${FB15K_INPUT:-${REPO}/data/fb15k-237/fb15k-237.tsv}"
PRIMEKG_KG_CSV="${PRIMEKG_KG_CSV:-${REPO}/data/kg.csv}"
ART="${ART:-${REPO}/results/slurm/artifacts}"
OUT_EMB="${OUT_EMB:-${REPO}/figures/umap/embeddings}"

# Encoder checkpoints (override with ENC_PRIMEKG_EDGE / ENC_FB15K_EDGE if filenames differ)
ENC_PRIMEKG_EDGE="${ENC_PRIMEKG_EDGE:-${ART}/hgt-primekg-edge-s42-3257009_0.encoder.pt}"
ENC_FB15K_EDGE="${ENC_FB15K_EDGE:-${ART}/hgt-fb15k237-edge-s42-3256272_2.encoder.pt}"

if [[ ! -f "${ENC_PRIMEKG_EDGE}" ]]; then
  echo "ERROR: PrimeKG HGT encoder not found: ${ENC_PRIMEKG_EDGE}" >&2
  exit 1
fi
if [[ ! -f "${ENC_FB15K_EDGE}" ]]; then
  echo "ERROR: FB15k HGT encoder not found: ${ENC_FB15K_EDGE}" >&2
  exit 1
fi

DEVICE="$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "[umap] Exporting HGT embeddings (device=${DEVICE})..."
python -m kgml_new.scripts.export_hgt_embeddings \
  --encoder "${ENC_PRIMEKG_EDGE}" \
  --output "${OUT_EMB}/hgt_primekg_edge.export.pt" \
  --input "${PRIMEKG_INPUT}" \
  --seed "${SEED}" \
  --split-protocol edge \
  --relation indication \
  --device "${DEVICE}"

python -m kgml_new.scripts.export_hgt_embeddings \
  --encoder "${ENC_FB15K_EDGE}" \
  --output "${OUT_EMB}/hgt_fb15k237_edge.export.pt" \
  --input "${FB15K_INPUT}" \
  --seed "${SEED}" \
  --split-protocol edge \
  --relation "/award/award_nominee/award_nominations./award/award_nomination/award_nominee" \
  --source-col source \
  --target-col target \
  --relation-col relation \
  --device "${DEVICE}"

MAN_FB="${MAN_FB:-${REPO}/figures/umap/manifest_fb15k237_edge.json}"
MAN_PK="${MAN_PK:-${REPO}/figures/umap/manifest_primekg_edge.json}"

MAX_NODES="${MAX_NODES:-12000}"

echo "[umap] FB15k-237 — relation diversity coloring (heterogeneous interaction counts)"
python -m kgml_new.scripts.umap_compare_methods \
  --manifest "${MAN_FB}" \
  --prepared-dataset-cache "${PKL_FB}" \
  --output-dir "${REPO}/figures/umap/out" \
  --stem "fb15k237_edge_umap" \
  --title "FB15k-237 edge split — methods" \
  --layout both \
  --max-nodes "${MAX_NODES}" \
  --seed "${SEED}" \
  --color-by labels \
  --labels-from-prepared "${PKL_FB}" \
  --label-mode relation_diversity \
  --save-npz

echo "[umap] PrimeKG — biological node-type coloring"
python -m kgml_new.scripts.umap_compare_methods \
  --manifest "${MAN_PK}" \
  --prepared-dataset-cache "${PKL_PRIME}" \
  --output-dir "${REPO}/figures/umap/out" \
  --stem "primekg_edge_umap" \
  --title "PrimeKG edge split — methods" \
  --layout both \
  --max-nodes "${MAX_NODES}" \
  --seed "${SEED}" \
  --color-by labels \
  --labels-from-kg-csv "${PRIMEKG_KG_CSV}" \
  --save-npz

echo "[umap] Done. Figures under figures/umap/out/"
