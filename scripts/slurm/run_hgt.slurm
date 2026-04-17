#!/bin/bash
# Typed KG link prediction with standalone HGT (Heterogeneous Graph Transformer).
#
#SBATCH --job-name=kgml-hgt
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
mkdir -p logs/slurm results/slurm

PYTHON="${PYTHON:-python}"
if [[ -x "${REPO}/.venv/bin/python" ]]; then
  PYTHON="${REPO}/.venv/bin/python"
fi

export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m kgml_new.scripts.run_hgt \
  --input data/kg.csv \
  --relation indication \
  --batch-size 1024 \
  --epochs 20 \
  --num-heads 4 \
  --num-layers 2 \
  --output "results/slurm/hgt-${SLURM_JOB_ID}.json"
