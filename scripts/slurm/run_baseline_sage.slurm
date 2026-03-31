#!/bin/bash
#SBATCH --job-name=kgml-sage
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new

.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --csv kg.csv \
  --epochs 200 \
  --output results/slurm/baseline_sage-${SLURM_JOB_ID}.json
