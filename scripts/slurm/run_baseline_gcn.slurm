#!/bin/bash
#SBATCH --job-name=kgml-gcn
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new
mkdir -p logs/slurm results/slurm
export PYTHONPATH=src

.venv/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_gcn \
  --csv data/kg.csv \
  --device cuda \
  --batch-size 512 \
  --epochs 100 \
  --output results/slurm/baseline_gcn-${SLURM_JOB_ID}.json
