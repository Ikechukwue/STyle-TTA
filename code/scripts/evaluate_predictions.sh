#!/bin/bash
# ============================================================================
# Evaluate Predictions (local)
# ============================================================================
# Computes metrics (accuracy, balanced accuracy, ECE) from TTA prediction
# JSON files. Also merges per-run files when needed.
#
# Usage:
#   bash scripts/evaluate_predictions.sh
#   bash scripts/evaluate_predictions.sh --results_dir ./results/ablation
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

echo "Collecting and merging results..."
python -m code.experiments.tta.collect_results --results_dir "$RESULTS_DIR"

echo "Evaluating predictions..."
python -m code.experiments.evaluate_predictions --results_dir "$RESULTS_DIR"
