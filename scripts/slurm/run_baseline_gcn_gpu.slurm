#!/bin/bash
#SBATCH --job-name=kgml-gcn-gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new
mkdir -p logs/slurm results/slurm
export PYTHONPATH=src

.venv_legacy_gpu/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_gcn \
  --csv data/kg.csv \
  --device cuda \
  --batch-size 128 \
  --epochs 100 \
  --output results/slurm/baseline_gcn_gpu-${SLURM_JOB_ID}.json
