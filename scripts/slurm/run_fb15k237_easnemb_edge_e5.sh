#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

REPO="/scratch/pioneer/users/sxk2517/kgml_new"
OUT_JSON="${OUT_JSON:?Set OUT_JSON to results path}"

cd "${REPO}"
mkdir -p logs/slurm results/slurm
source .venv/bin/activate
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

exec python -m kgml_new.scripts.run_gpu_method \
  --require-cuda \
  --method edge_aware_sage_node_emb \
  --input "${REPO}/data/fb15k-237/fb15k-237.tsv" \
  --input-format auto \
  --split-protocol edge \
  --seed 42 \
  --in-dim 128 \
  --decoder dot \
  --negatives-per-pos 20 \
  --prepared-dataset-cache "${REPO}/cache/fb15k237-prepared_edge_s42_d128.pkl" \
  --epochs 100 \
  --semantic \
  --strict-semantic \
  --embedding-model e5 \
  --edge-relation-mode concat \
  --edge-dim 128 \
  --semantic-cache "${REPO}/cache/fb15k237-relations-e5.pt" \
  --node-embeddings-path "${REPO}/cache/fb15k237-node-embeddings-e5.pt" \
  --log-file "${REPO}/logs/slurm/run-kgml-easnemb-fb15k237-edge-e5-%j.log" \
  --log-level INFO \
  --output "${OUT_JSON}"
