#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

REPO="/scratch/pioneer/users/sxk2517/kgml_new"
OUT_JSON="${OUT_JSON:?Set OUT_JSON to results path}"
PRIMEKG_INPUT="${PRIMEKG_INPUT:-${REPO}/data/primekg.tsv}"

cd "${REPO}"
mkdir -p logs/slurm results/slurm
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method edge_aware_sage_node_emb \
  --input "${PRIMEKG_INPUT}" \
  --input-format auto \
  --split-protocol node \
  --seed 42 \
  --in-dim 128 \
  --decoder dot \
  --negatives-per-pos 20 \
  --prepared-dataset-cache "${REPO}/cache/primekg-prepared_node_s42_d128.pkl" \
  --epochs 100 \
  --semantic \
  --strict-semantic \
  --embedding-model e5 \
  --edge-relation-mode concat \
  --edge-dim 128 \
  --semantic-cache "${REPO}/cache/primekg-relations-e5.pt" \
  --node-embeddings-path "${REPO}/cache/primekg-node-embeddings-e5.pt" \
  --log-file "${REPO}/logs/slurm/run-kgml-easnemb-primekg-node-e5-${SLURM_JOB_ID:-local}.log" \
  --log-level INFO \
  --output "${OUT_JSON}"
