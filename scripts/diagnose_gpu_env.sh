#!/usr/bin/env bash
# Print a one-page report: NVIDIA driver/GPU vs Python torch vs PyG extension .so load.
# Run on the *same* node class you use for training (GPU partition), with your venv active.
#
# Usage:
#   source .venv/bin/activate
#   ./scripts/diagnose_gpu_env.sh
#
set -u

echo "========== Host / GPU =========="
echo "hostname: $(hostname 2>/dev/null || true)"
echo "date: $(date -Is 2>/dev/null || date)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "--- nvidia-smi (driver + GPU) ---"
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null || nvidia-smi
  echo ""
  echo "--- nvidia-smi header (CUDA version line) ---"
  nvidia-smi 2>/dev/null | head -n 3
else
  echo "nvidia-smi: NOT FOUND (not a GPU node, or driver not loaded)"
fi

echo ""
echo "--- nvcc (optional; may be absent on compute nodes) ---"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not in PATH (normal on many clusters)"
fi

echo ""
echo "========== Python / venv =========="
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<not set>}"
echo "python: $(command -v python 2>/dev/null || echo missing)"

echo ""
echo "--- torch / CUDA (Python) ---"
python - <<'PY'
import sys

print("executable:", sys.executable)
try:
    import torch
except Exception as e:
    print("torch import FAILED:", e)
    raise SystemExit(1)

print("torch.__version__:", torch.__version__)
print("torch.version.cuda (build):", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.backends.cudnn.version():", torch.backends.cudnn.version() if torch.cuda.is_available() else "n/a")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print("device_count:", n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  capability={p.major}.{p.minor}  total_mem_GiB={p.total_memory / 2**30:.2f}")
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.randn(1024, 1024, device="cuda")
        z = (x @ y).sum().item()
        print("cuda matmul smoke: OK (scalar)", float(z))
    except Exception as e:
        print("cuda matmul smoke: FAILED:", e)
else:
    print("CUDA not available from PyTorch — common causes:")
    print("  - CPU-only torch wheel installed")
    print("  - torch wheel CUDA tag newer than driver supports")
    print("  - No GPU on this node / MIG / cgroup device deny")

print("")
print("--- PyG extension modules (optional) ---")
for mod in ("pyg_lib", "torch_scatter", "torch_sparse"):
    try:
        __import__(mod)
        print(mod + ": import OK")
    except Exception as e:
        print(mod + ": import FAILED:", e)
PY

echo ""
echo "========== Interpretation (short) =========="
echo "1) nvidia-smi must show GPUs on a GPU node; if not, you are on login/shared CPU."
echo "2) torch.version.cuda is the CUDA toolkit PyTorch was built for (e.g. 12.8 for +cu128)."
echo "   Driver from nvidia-smi must be new enough for that build (see pytorch.org compatibility table)."
echo "3) If torch.cuda.is_available() is False on a GPU node, reinstall torch from your cluster's"
echo "   documented wheel index (or pytorch.org) for the correct cuXXX tag, then reinstall PyG binaries."
echo "4) PyG scatter/sparse failures often mean extensions were installed for a *different* torch build;"
echo "   reinstall after changing torch: pip uninstall pyg_lib torch_scatter torch_sparse -y && ./scripts/install_pyg_extensions.sh"
echo "   (if that script is not in tree, use the PyG wheel URL matching torch.__version__)."
echo "5) GLIBC_2.29 errors loading libpyg.so on RHEL/Rocky 8: pip wheels need newer glibc — run"
echo "   module load GCCcore/12.2.0 CUDA/12.2.0 && source .venv/bin/activate && ./scripts/install_pyg_extensions_el8.sh"
