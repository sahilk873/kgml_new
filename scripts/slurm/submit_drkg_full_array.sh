#!/usr/bin/env bash
# Submit the DRKG full experiment array from repo root. Edit #SBATCH lines inside the
# .slurm file for your cluster (partition, account, qos, mem, time).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p logs/slurm results/slurm
JOBID="$(sbatch --parsable scripts/slurm/drkg_full_experiments_array.slurm)"
echo "Submitted job array: $JOBID"
echo "Monitor with: ./scripts/slurm/monitor_slurm_jobs.sh $JOBID"
