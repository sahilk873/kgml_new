#!/bin/bash
#SBATCH --job-name=kgml-node2vec
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new

.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method node2vec \
  --csv kg.csv \
  --epochs 20 \
  --output results/slurm/node2vec-${SLURM_JOB_ID}.json
