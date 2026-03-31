#!/bin/bash
#SBATCH --job-name=kgml-txgnn
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new

.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --csv kg.csv \
  --relation indication \
  --epochs 20 \
  --output results/slurm/txgnn-${SLURM_JOB_ID}.json
