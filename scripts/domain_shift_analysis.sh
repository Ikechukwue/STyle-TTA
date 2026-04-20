#!/bin/bash
# ============================================================================
# Run Domain Shift Analysis (local)
# ============================================================================
# Quantifies the domain gap between ImageNet-1k and its shifted variants
# using texture, colour, and feature-space metrics.
#
# Usage:
#   bash scripts/domain_shift_analysis.sh
#   bash scripts/domain_shift_analysis.sh --target_split test_a
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TARGET_SPLIT="test_r"

while [[ $# -gt 0 ]]; do
    case $1 in
        --target_split) TARGET_SPLIT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Domain shift analysis: imagenet/val → imagenet/${TARGET_SPLIT}"

python -m experiments.domain_shift_analysis \
    --source_dataset imagenet --source_split val \
    --target_dataset imagenet --target_split "$TARGET_SPLIT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_PATH/domain_shift_analysis" \
    --n_samples 500 --seed 42
