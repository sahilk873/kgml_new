#!/bin/bash
#SBATCH --job-name=kgml-node2vec-primekg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

REPO=/scratch/pioneer/users/sxk2517/kgml_new
cd "$REPO"
mkdir -p logs/slurm results/slurm/artifacts
export PYTHONPATH=src

.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method node2vec \
  --input data/kg.csv \
  --device cuda \
  --batch-size 512 \
  --epochs 100 \
  --embed-dim 64 \
  --walk-length 20 \
  --context-size 15 \
  --walks-per-node 4 \
  --split-protocol edge \
  --artifacts-dir results/slurm/artifacts \
  --output results/slurm/primekg_node2vec-${SLURM_JOB_ID}.json