#!/usr/bin/env bash
# Watch Slurm job(s) until completion; print accounting; on failure show log hints.
#
# Usage:
#   ./scripts/slurm/monitor_slurm_jobs.sh JOBID           # array or single (drkg-full-* logs)
#   ./scripts/slurm/monitor_slurm_jobs.sh JOBID --poll 30 # seconds between squeue checks
#
# Single-job logs (e.g. drkg-gcn-12345.out from drkg_baseline_gcn.slurm):
#   LOG_PREFIX=drkg-gcn ./scripts/slurm/monitor_slurm_jobs.sh 12345
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 JOBID [--poll SECONDS]" >&2
  exit 1
fi

JOBID="$1"
POLL=60
if [[ "${2:-}" == "--poll" && -n "${3:-}" ]]; then
  POLL="$3"
fi

echo "Watching job $JOBID (poll ${POLL}s)..."
while squeue -j "$JOBID" -h 2>/dev/null | grep -q .; do
  squeue -j "$JOBID" -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" 2>/dev/null || true
  sleep "$POLL"
done

echo ""
echo "=== sacct (state / exit / resources) ==="
sacct -j "$JOBID" --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,AllocTRES%40 2>/dev/null || echo "sacct unavailable"

echo ""
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${LOG_PREFIX:-}" ]]; then
  echo "=== log files (single job: ${LOG_PREFIX}-${JOBID}) ==="
  ls -lt "${REPO}/logs/slurm/${LOG_PREFIX}-${JOBID}".out 2>/dev/null || echo "No logs/slurm/${LOG_PREFIX}-${JOBID}.out yet."
else
  echo "=== log files (latest array tasks) ==="
  ls -lt "${REPO}/logs/slurm"/drkg-full-"${JOBID}"_*.out 2>/dev/null | head -5 || echo "No matching logs/slurm/drkg-full-${JOBID}_*.out yet."
fi

FAILED="$(sacct -j "$JOBID" -X -n -o State 2>/dev/null | grep -E 'FAIL|CANCEL|TIMEOUT|OUT_OF_ME|NODE_FAIL' || true)"
if [[ -n "$FAILED" ]]; then
  echo ""
  echo "Some tasks may have failed. Tail stderr:"
  if [[ -n "${LOG_PREFIX:-}" ]]; then
    f="${REPO}/logs/slurm/${LOG_PREFIX}-${JOBID}.err"
    if [[ -f "$f" ]]; then
      echo "--- $f (last 40 lines) ---"
      tail -n 40 "$f"
    fi
  else
    for f in "${REPO}/logs/slurm"/drkg-full-"${JOBID}"_*.err; do
      [[ -f "$f" ]] || continue
      echo "--- $f (last 40 lines) ---"
      tail -n 40 "$f"
    done
  fi
  echo ""
  echo "Python run logs (if --log-file was used):"
  if [[ -n "${LOG_PREFIX:-}" ]]; then
    ls -1 "${REPO}/logs/slurm"/run-drkg-*-"${JOBID}".log 2>/dev/null | while read -r lf; do
      echo "--- $lf (last 30 lines) ---"
      tail -n 30 "$lf"
    done || true
  else
    ls -1 "${REPO}/logs/slurm"/run-drkg-"${JOBID}"_*.log 2>/dev/null | while read -r lf; do
      echo "--- $lf (last 30 lines) ---"
      tail -n 30 "$lf"
    done || true
  fi
fi
