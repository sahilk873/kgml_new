#!/bin/bash
# Setup script for KGML on HPC
# Usage: ./scripts/setup_hpc.sh

set -e

echo "=== KGML HPC Setup Script ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Create data directory if it doesn't exist
mkdir -p "$PROJECT_DIR/data"

# Download PrimeKG if not present
if [ -f "$PROJECT_DIR/kg.csv" ]; then
    echo "PrimeKG data already exists at $PROJECT_DIR/kg.csv"
else
    echo "Downloading PrimeKG from Harvard Dataverse..."
    wget -O "$PROJECT_DIR/kg.csv" https://dataverse.harvard.edu/api/access/datafile/6180620
    echo "Downloaded PrimeKG to $PROJECT_DIR/kg.csv"
fi

# Check if virtual environment exists
if [ -d "$PROJECT_DIR/.venv" ]; then
    echo "Virtual environment already exists"
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source "$PROJECT_DIR/.venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install -e .

# Install optional dependencies for semantic embeddings (optional)
read -p "Install OpenAI for semantic embeddings? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[semantic]"
fi

# Install dev dependencies
echo "Installing dev dependencies (pytest)..."
pip install -e ".[dev]"

# Install pyg-lib for neighbor sampling (required for batched training)
echo "Installing pyg-lib for neighbor sampling..."
pip install pyg-lib -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html || true

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run a quick test:"
echo "  python -m pytest tests/ -v"
echo ""
echo "To train GraphSAGE:"
echo "  python -m kgml_new.scripts.run_link_prediction --graph data/graph.pkl --model sage --epochs 20"
echo ""
