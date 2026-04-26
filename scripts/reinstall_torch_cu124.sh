#!/usr/bin/env bash
# Reinstall PyTorch + PyG extensions for CUDA 12.4 wheels (typical match for nvidia-smi "CUDA Version: 12.x").
# PyTorch +cu130 requires a newer driver than many 12.8-capable nodes; this downgrades the pip stack.
#
# Run once on a login or GPU node with the project venv active:
#   source .venv/bin/activate
#   ./scripts/reinstall_torch_cu124.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

echo "Uninstalling old torch / PyG CUDA wheels..."
pip uninstall -y torch torchvision torchaudio pyg_lib torch_scatter torch_sparse 2>/dev/null || true

echo "Installing PyTorch (cu124 index)..."
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124

TORCH_VER="$(python -c "import torch; print(torch.__version__)")"
echo "Installed torch ${TORCH_VER}"

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}.html"
echo "Installing PyG binaries from ${PYG_URL}"
pip install torch_scatter torch_sparse -f "${PYG_URL}"
if [[ "${INSTALL_PYG_LIB:-0}" == "1" ]]; then
  pip install pyg_lib -f "${PYG_URL}" || {
    echo "WARNING: pyg_lib install failed (likely old glibc); continuing with torch_sparse backend." >&2
  }
fi
echo ""
echo "If imports fail with GLIBC_2.29 on RHEL/Rocky 8, keep pyg_lib uninstalled or build from source:"
echo "  module load GCCcore/12.2.0 CUDA/12.2.0   # GCC major must be <=12 for CUDA 12.2 nvcc"
echo "  source .venv/bin/activate && ./scripts/install_pyg_extensions_el8.sh"
echo "By default this script now installs torch_scatter + torch_sparse only (enough for LinkNeighborLoader)."

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY

echo "Done. Re-run: python -c \"import torch; print(torch.cuda.is_available())\" on a GPU node."
