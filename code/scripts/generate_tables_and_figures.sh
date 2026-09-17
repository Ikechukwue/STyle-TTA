#!/bin/bash
# ============================================================================
# Generate LaTeX Tables and Figures (local)
# ============================================================================
# Produces all thesis tables and figures from the collected results.
#
# Usage:
#   bash scripts/generate_tables_and_figures.sh
#   bash scripts/generate_tables_and_figures.sh --results_dir ./results
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

RESULTS_DIR="$OUTPUT_PATH"

while [[ $# -gt 0 ]]; do
    case $1 in
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Generating LaTeX tables..."
python -m code.experiments.thesis.reporting.generate_latex_tables \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/tables"

echo "Generating figures..."
python -m code.experiments.thesis.reporting.visualize_results \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/figures"

echo "Done. Tables: $RESULTS_DIR/tables/ | Figures: $RESULTS_DIR/figures/"
