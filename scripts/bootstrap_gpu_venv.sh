#!/usr/bin/env bash
# Create or refresh kgml_new/.venv with CUDA PyTorch + torch_geometric + torch_sparse/pyg_lib
# so HGT LinkNeighborLoader works (neighbor_sampler_backend_available() == True).
#
# Usage (from repo root):
#   RECREATE=1 bash scripts/bootstrap_gpu_venv.sh
#
# Override CUDA wheel tag if your cluster uses another toolkit (must match GPU driver):
#   CUDA_TAG=cu126 RECREATE=1 bash scripts/bootstrap_gpu_venv.sh
#
# Use a clean base interpreter (not another venv's python):
#   SYS_PYTHON=/usr/bin/python3 bash scripts/bootstrap_gpu_venv.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"

SYS_PYTHON="${SYS_PYTHON:-}"
if [[ -z "${SYS_PYTHON}" ]] || [[ ! -x "${SYS_PYTHON}" ]]; then
  # Prefer HPC EasyBuild Python if present (matches typical worker nodes).
  EB_PY="/usr/local/easybuild_allnodes/software/Python/3.11.5-GCCcore-13.2.0/bin/python3"
  if [[ -x "${EB_PY}" ]]; then
    SYS_PYTHON="${EB_PY}"
  else
    SYS_PYTHON="$(command -v python3)"
  fi
fi

CUDA_TAG="${CUDA_TAG:-cu124}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/${CUDA_TAG}}"
RECREATE="${RECREATE:-0}"

echo "Using SYS_PYTHON=${SYS_PYTHON}"
"${SYS_PYTHON}" --version

if [[ "${RECREATE}" == "1" ]] || [[ ! -x "${REPO}/.venv/bin/python" ]]; then
  echo "Creating venv at ${REPO}/.venv ..."
  "${SYS_PYTHON}" -m venv --clear "${REPO}/.venv"
fi

VENV_PY="${REPO}/.venv/bin/python"
PIP="${REPO}/.venv/bin/pip"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: ${VENV_PY} not executable after venv creation." >&2
  exit 1
fi

"${PIP}" install -U pip setuptools wheel

echo "Installing CUDA PyTorch from ${TORCH_INDEX} (avoid CPU-only torch from PyPI) ..."
"${PIP}" install --index-url "${TORCH_INDEX}" "torch>=2.6,<3" torchvision

echo "Installing project + dev/tuning extras ..."
"${PIP}" install -e "${REPO}[dev,tuning]"

TORCH_VER="$("${VENV_PY}" -c "import torch; print(torch.__version__)")"
PYG_WHL="https://data.pyg.org/whl/torch-${TORCH_VER}.html"
# torch_scatter is required for torch_sparse. Do not install pyg_lib on RHEL8 / glibc<2.29:
# prebuilt libpyg.so needs GLIBC_2.29; its failed import can also break torch_sparse detection
# in torch_geometric. Neighbor sampling only needs torch_sparse (see hgt_train.neighbor_sampler_backend_available).
echo "Installing torch_scatter + torch_sparse from ${PYG_WHL} ..."
"${PIP}" install torch_scatter torch_sparse -f "${PYG_WHL}"
if [[ "${INSTALL_PYG_LIB:-0}" == "1" ]]; then
  echo "INSTALL_PYG_LIB=1: also installing pyg_lib (requires glibc 2.29+ for prebuilt wheel) ..."
  "${PIP}" install pyg_lib -f "${PYG_WHL}" || {
    echo "WARNING: pyg_lib install/import may fail on old glibc; torch_sparse alone is enough for HGT neighbor sampling." >&2
  }
fi

export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${VENV_PY}" -c "
import torch
from kgml_new.training.hgt_train import neighbor_sampler_backend_available

assert neighbor_sampler_backend_available(), (
    'neighbor_sampler_backend_available() is False — check torch_sparse/pyg_lib wheels'
)
print('torch:', torch.__version__)
print('cuda built:', torch.version.cuda)
print('neighbor_sampler_backend_available:', neighbor_sampler_backend_available())
"

echo "Done. Activate with: source ${REPO}/.venv/bin/activate"
