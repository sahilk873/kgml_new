#!/bin/bash
# Setup script for KGML on HPC
# Usage:
#   ./scripts/setup_hpc.sh
#   ./scripts/setup_hpc.sh --dataset-url https://... --dataset-path data/my_kg.csv --venv .venv312

set -euo pipefail

echo "=== KGML HPC Setup Script ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV_PATH=".venv"
DATASET_PATH="kg.csv"
DATASET_URL="https://dataverse.harvard.edu/api/access/datafile/6180620"
INSTALL_SEMANTIC="ask"
INSTALL_DEV="yes"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --dataset-url)
            DATASET_URL="$2"
            shift 2
            ;;
        --with-semantic)
            INSTALL_SEMANTIC="yes"
            shift
            ;;
        --without-semantic)
            INSTALL_SEMANTIC="no"
            shift
            ;;
        --without-dev)
            INSTALL_DEV="no"
            shift
            ;;
        --help|-h)
            cat <<'EOF'
Usage: ./scripts/setup_hpc.sh [options]

Options:
  --venv PATH            Virtualenv path. Default: .venv
  --dataset-path PATH    Local CSV path to ensure exists. Default: kg.csv
  --dataset-url URL      Optional URL to download the CSV if missing.
  --with-semantic        Install semantic extras without prompting.
  --without-semantic     Skip semantic extras without prompting.
  --without-dev          Skip dev extras.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$PROJECT_DIR/data"

if [[ -f "$DATASET_PATH" ]]; then
    echo "Dataset already exists at $DATASET_PATH"
elif [[ -n "$DATASET_URL" ]]; then
    echo "Downloading dataset to $DATASET_PATH"
    mkdir -p "$(dirname "$DATASET_PATH")"
    curl -L "$DATASET_URL" -o "$DATASET_PATH"
else
    echo "Dataset not found at $DATASET_PATH and no download URL provided." >&2
    exit 1
fi

if [[ -d "$VENV_PATH" ]]; then
    echo "Virtual environment already exists at $VENV_PATH"
else
    echo "Creating virtual environment at $VENV_PATH"
    python3.12 -m venv "$VENV_PATH" 2>/dev/null || python3 -m venv "$VENV_PATH"
fi

source "$PROJECT_DIR/$VENV_PATH/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing core package..."
python -m pip install -e .

if [[ "$INSTALL_DEV" == "yes" ]]; then
    echo "Installing dev dependencies..."
    python -m pip install -e ".[dev]"
fi

if [[ "$INSTALL_SEMANTIC" == "ask" ]]; then
    read -r -p "Install OpenAI/python-dotenv semantic extras? (y/n) " reply
    if [[ "$reply" =~ ^[Yy]$ ]]; then
        INSTALL_SEMANTIC="yes"
    else
        INSTALL_SEMANTIC="no"
    fi
fi

if [[ "$INSTALL_SEMANTIC" == "yes" ]]; then
    echo "Installing semantic extras..."
    python -m pip install -e ".[semantic]"
fi

echo "Installing PyTorch Geometric neighbor-sampling backends if available..."
PYTORCH_VERSION="$(python - <<'PY'
import torch
print(torch.__version__.split('+')[0])
PY
)"
PYG_WHL_SUFFIX="$(python - <<'PY'
import torch

local_suffix = torch.__version__.split('+', 1)[1] if '+' in torch.__version__ else None
if local_suffix:
    print(local_suffix)
elif torch.version.cuda:
    print("cu" + torch.version.cuda.replace(".", ""))
else:
    print("cpu")
PY
)"
python -m pip install torch-geometric || true
python -m pip install pyg-lib torch-sparse -f "https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${PYG_WHL_SUFFIX}.html" || true

echo ""
echo "=== Setup Complete ==="
echo "Activate with:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Quick validation:"
echo "  pytest tests/test_graph.py tests/test_training.py tests/test_models.py -q"
echo ""
echo "Prepare a CSV into a graph pickle:"
echo "  ./scripts/prepare_data.sh --input-csv $DATASET_PATH --output-pkl data/graph.pkl"
