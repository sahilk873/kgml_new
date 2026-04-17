#!/bin/bash
#SBATCH --job-name=kgml-sage-legacy
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

cd /nas/longleaf/home/sahilk/kgml_new

.venv_legacy_gpu/bin/python -m kgml_new.scripts.run_gpu_method \
  --method baseline_sage \
  --csv data/kg.csv \
  --epochs 200 \
  --output results/slurm/baseline_sage_legacy-${SLURM_JOB_ID}.json
