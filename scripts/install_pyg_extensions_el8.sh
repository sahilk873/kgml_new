#!/usr/bin/env bash
# Build PyG neighbor-sampling extensions on RHEL/Rocky 8 (glibc 2.28).
#
# Official PyG wheels link against glibc >= 2.29 and fail with:
#   version `GLIBC_2.29' not found (required by libpyg.so)
#
# This script rebuilds pyg-lib from source with the cluster CUDA toolkit and a
# GCC version nvcc accepts (<= 12 for CUDA 12.2), and puts matching GCC binutils
# first on PATH so extension builds do not pick up an old system `as`.
#
# Prereqs (typical EasyBuild — adjust module names for your site):
#   module load GCCcore/12.2.0 CUDA/12.2.0
#
# Usage (repo root, venv active or .venv present):
#   module load GCCcore/12.2.0 CUDA/12.2.0
#   source .venv/bin/activate
#   ./scripts/install_pyg_extensions_el8.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not in PATH. Load a CUDA module, e.g.:" >&2
  echo "  module load CUDA/12.2.0" >&2
  exit 1
fi

GCC_VER="$(gcc -dumpversion 2>/dev/null | cut -d. -f1)"
if [[ "${GCC_VER:-0}" -gt 12 ]]; then
  echo "ERROR: gcc major version is ${GCC_VER:-?}; nvcc 12.2 rejects GCC > 12." >&2
  echo "  module load GCCcore/12.2.0   # or another GCC <= 12 toolchain" >&2
  exit 1
fi

# Prefer binutils from the same toolchain as g++ (avoids: as: unrecognized option '--gdwarf-5').
GCC_PREFIX="$(dirname "$(command -v gcc)")"
export PATH="${GCC_PREFIX}:${PATH}"

TORCH_VER="$(python -c "import torch; print(torch.__version__)")"
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}.html"

echo "=== PyG EL8-friendly install ==="
echo "torch: ${TORCH_VER}"
echo "CUDA_HOME: ${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
echo "gcc: $(command -v gcc) ($(gcc --version | head -1))"
echo "as:  $(command -v as)"

echo ""
echo "Uninstalling pip pyg_lib / torch_scatter / torch_sparse (if present)..."
pip uninstall -y pyg_lib torch_scatter torch_sparse 2>/dev/null || true

pip install -q --upgrade pip setuptools wheel ninja cmake

echo ""
echo "Building pyg-lib from source (tag 0.5.0, matches common torch 2.6 + cu124 stacks)..."
export MAX_JOBS="${MAX_JOBS:-8}"
pip install --no-build-isolation "git+https://github.com/pyg-team/pyg-lib.git@0.5.0"

echo ""
echo "Installing torch_scatter / torch_sparse wheels from PyG index (match torch)..."
pip install torch_scatter torch_sparse -f "${PYG_URL}"

verify() {
  python - <<'PY'
import pyg_lib
import torch_scatter
import torch_sparse
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE

print("pyg_lib, torch_scatter, torch_sparse: OK")
print("WITH_PYG_LIB:", WITH_PYG_LIB, "WITH_TORCH_SPARSE:", WITH_TORCH_SPARSE)
PY
}

echo ""
echo "Verifying imports..."
if ! verify; then
  echo ""
  echo "Wheel-based scatter/sparse failed (often glibc on EL8). Building from source..."
  pip uninstall -y torch_scatter torch_sparse 2>/dev/null || true
  pip install --no-build-isolation "git+https://github.com/rusty1s/pytorch_scatter.git@2.1.2"
  pip install --no-build-isolation "git+https://github.com/rusty1s/pytorch_sparse.git@0.6.18"
  echo ""
  echo "Re-verifying..."
  verify
fi

echo ""
echo "Done. Re-run on a GPU node: ./scripts/diagnose_gpu_env.sh"
