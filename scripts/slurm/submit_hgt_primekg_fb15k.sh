#!/usr/bin/env bash
# Submit GPU array for HGT on PrimeKG × FB15k237 × edge/node splits.
#
# Usage:
#   ./scripts/slurm/submit_hgt_primekg_fb15k.sh
#
# FB15k requires a relation name:
#   export HGT_RELATION_FB15K237='/location/country/form_of_government'
#
# Example (from repo root):
#   export HGT_RELATION_FB15K237='/people/person/nationality'
#   ./scripts/slurm/submit_hgt_primekg_fb15k.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

if [[ -z "${HGT_RELATION_FB15K237:-}" ]] && [[ -z "${SKIP_FB15K_REL_CHECK:-}" ]]; then
  echo "NOTE: Export HGT_RELATION_FB15K237 to a relation string present in FB15k (column 2)." >&2
  echo "      Example: export HGT_RELATION_FB15K237='/award/award_category/winners./award/award_honor/award_winner'" >&2
  echo "      Or set SKIP_FB15K_REL_CHECK=1 to submit anyway (the fb15k array tasks will fail at runtime)." >&2
fi

JOB="$(sbatch --parsable "${REPO}/scripts/slurm/hgt_primekg_fb15k_split_array.slurm")"
echo "Submitted ${JOB}"
echo "Logs: ${REPO}/logs/slurm/kgml-hgt-primekg-fb15k-${JOB}_*.out"
