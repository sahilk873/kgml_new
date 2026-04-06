#!/usr/bin/env bash
# Install PyG binary wheels (pyg_lib, torch_scatter, torch_sparse) matching the *current* torch build.
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
python -m pip install pyg_lib torch_scatter torch_sparse -f "$PYG_WHL_URL"
echo "Done."
