#!/bin/bash
#SBATCH --job-name=kgml-txgnn
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new

.venv/bin/python -m kgml_new.scripts.run_txgnn \
  --csv kg.csv \
  --max-edges 5000 \
  --relation indication \
  --epochs 3 \
  --output results/slurm/txgnn-${SLURM_JOB_ID}.json
