#!/bin/bash
# Convert a KG CSV into a NetworkX pickle for training.
# Usage:
#   ./scripts/prepare_data.sh
#   ./scripts/prepare_data.sh --input-csv data/my_kg.csv --output-pkl data/my_kg.pkl \
#       --source-col src --target-col dst --relation-col rel --source-type-col src_type --target-type-col dst_type

set -euo pipefail

echo "=== Converting KG CSV to NetworkX pickle ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VENV_PATH=".venv"
INPUT_CSV="kg.csv"
OUTPUT_PKL="data/graph.pkl"
SOURCE_COL="x_name"
TARGET_COL="y_name"
RELATION_COL="relation"
SOURCE_TYPE_COL="x_type"
TARGET_TYPE_COL="y_type"
MAX_EDGES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --input-csv)
            INPUT_CSV="$2"
            shift 2
            ;;
        --output-pkl)
            OUTPUT_PKL="$2"
            shift 2
            ;;
        --source-col)
            SOURCE_COL="$2"
            shift 2
            ;;
        --target-col)
            TARGET_COL="$2"
            shift 2
            ;;
        --relation-col)
            RELATION_COL="$2"
            shift 2
            ;;
        --source-type-col)
            SOURCE_TYPE_COL="$2"
            shift 2
            ;;
        --target-type-col)
            TARGET_TYPE_COL="$2"
            shift 2
            ;;
        --max-edges)
            MAX_EDGES="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'EOF'
Usage: ./scripts/prepare_data.sh [options]

Options:
  --venv PATH               Virtualenv path. Default: .venv
  --input-csv PATH          Input CSV. Default: kg.csv
  --output-pkl PATH         Output pickle. Default: data/graph.pkl
  --source-col NAME         Source node column.
  --target-col NAME         Target node column.
  --relation-col NAME       Edge relation column.
  --source-type-col NAME    Optional source node type column.
  --target-type-col NAME    Optional target node type column.
  --max-edges N             Limit rows before conversion.
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Error: input CSV not found at $INPUT_CSV" >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d "$PROJECT_DIR/$VENV_PATH" ]]; then
        echo "Activating virtual environment at $VENV_PATH"
        source "$PROJECT_DIR/$VENV_PATH/bin/activate"
    fi
fi

mkdir -p "$(dirname "$OUTPUT_PKL")"

KGML_SOURCE_COL="$SOURCE_COL" \
KGML_TARGET_COL="$TARGET_COL" \
KGML_RELATION_COL="$RELATION_COL" \
KGML_SOURCE_TYPE_COL="$SOURCE_TYPE_COL" \
KGML_TARGET_TYPE_COL="$TARGET_TYPE_COL" \
KGML_INPUT_CSV="$INPUT_CSV" \
KGML_OUTPUT_PKL="$OUTPUT_PKL" \
KGML_MAX_EDGES="$MAX_EDGES" \
python - <<'PY'
import os
from pathlib import Path
import pickle

from kgml_new.data.loaders import GraphCSVSpec, load_graph_csv

spec = GraphCSVSpec(
    source_col=os.environ["KGML_SOURCE_COL"],
    target_col=os.environ["KGML_TARGET_COL"],
    relation_col=os.environ["KGML_RELATION_COL"],
    source_type_col=os.environ["KGML_SOURCE_TYPE_COL"] or None,
    target_type_col=os.environ["KGML_TARGET_TYPE_COL"] or None,
)

max_edges = None
if os.environ["KGML_MAX_EDGES"]:
    max_edges = int(os.environ["KGML_MAX_EDGES"])

csv_path = Path(os.environ["KGML_INPUT_CSV"])
output_path = Path(os.environ["KGML_OUTPUT_PKL"])

print(f"Loading CSV from {csv_path}")
graph = load_graph_csv(csv_path, spec=spec, max_edges=max_edges)
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

with output_path.open("wb") as f:
    pickle.dump(graph, f)

print(f"Saved pickle to {output_path}")
PY

echo ""
echo "=== Conversion Complete ==="
echo "You can now run training with:"
echo "  python -m kgml_new.scripts.run_link_prediction --graph $OUTPUT_PKL --model sage --epochs 20"
