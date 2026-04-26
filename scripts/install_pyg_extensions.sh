#!/usr/bin/env bash
# Install PyG binary wheels matching the *current* torch build.
# Default install is torch_scatter + torch_sparse (sufficient for neighbor sampling).
# pyg_lib is optional and may fail on glibc<2.29 (e.g. RHEL8).
# Re-run after changing PyTorch CUDA wheels on a GPU node.
#
# Usage (repo root, venv active):
#   ./scripts/install_pyg_extensions.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

PYG_WHL_URL="$(python - <<'PY'
import torch

v = torch.__version__.strip()
if "+" not in v:
    v = f"{v}+cpu"
print(f"https://data.pyg.org/whl/torch-{v}.html")
PY
)"

echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "PyG wheel index: $PYG_WHL_URL"
python -m pip install torch_scatter torch_sparse -f "$PYG_WHL_URL"
if [[ "${INSTALL_PYG_LIB:-0}" == "1" ]]; then
  python -m pip install pyg_lib -f "$PYG_WHL_URL" || {
    echo "WARNING: pyg_lib install failed (likely old glibc). Continuing with torch_sparse backend." >&2
  }
fi
python - <<'PY'
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
print("WITH_PYG_LIB:", WITH_PYG_LIB, "WITH_TORCH_SPARSE:", WITH_TORCH_SPARSE)
PY
echo "Done."
