#!/bin/bash
#SBATCH --job-name=kgml-sage
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new

.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --csv kg.csv \
  --max-edges 5000 \
  --epochs 3 \
  --output results/slurm/baseline_sage-${SLURM_JOB_ID}.json
