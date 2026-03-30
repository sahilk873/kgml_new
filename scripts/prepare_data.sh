#!/bin/bash
# Convert PrimeKG CSV to NetworkX pickle for training
# Usage: ./scripts/prepare_data.sh

set -e

echo "=== Converting PrimeKG CSV to NetworkX pickle ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Check if CSV exists
if [ ! -f "$PROJECT_DIR/kg.csv" ]; then
    echo "Error: kg.csv not found. Run setup_hpc.sh first."
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# Create data directory
mkdir -p "$PROJECT_DIR/data"

# Run Python script to convert
python3 -c "
import pandas as pd
import networkx as nx
from pathlib import Path

print('Loading CSV...')
df = pd.read_csv('kg.csv', low_memory=False)
print(f'Loaded {len(df)} edges')

print('Building graph...')
g = nx.Graph()

for _, row in df.iterrows():
    src = str(row['x_name'])
    tgt = str(row['y_name'])
    rel = str(row['relation'])
    
    g.add_node(src, node_type=row.get('x_type', 'unknown'))
    g.add_node(tgt, node_type=row.get('y_type', 'unknown'))
    g.add_edge(src, tgt, relationship=rel)

print(f'Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges')

# Save as pickle
output_path = Path('data/primekg.pkl')
import pickle
with open(output_path, 'wb') as f:
    pickle.dump(g, f)

print(f'Saved to {output_path}')
"

echo ""
echo "=== Conversion Complete! ==="
echo "You can now run training with:"
echo "  python -m kgml_new.scripts.run_link_prediction --graph data/primekg.pkl --model sage --epochs 20"
