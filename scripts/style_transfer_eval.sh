#!/bin/bash
# ============================================================================
# Run Style Transfer Method Evaluation (local)
# ============================================================================
# Evaluates all style transfer methods on ImageNet-R (content) vs ImageNet-1k
# (style) using 20 content × 40 style = 800 stylized images per method.
#
# Usage:
#   bash scripts/style_transfer_eval.sh
#   bash scripts/style_transfer_eval.sh --methods styleid diffstyle
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

python -m experiments.style_transfer_evaluation \
    --data_path "$DATA_PATH" \
    --weights_dir "$WEIGHTS_DIR" \
    --output_dir "$OUTPUT_PATH/style_transfer_eval" \
    --content_dataset imagenet --content_split test_r \
    --style_dataset imagenet --style_split train@test_r \
    --n_content 3 --n_style 3 \
    --input_size 256 --seed 42 \
    "$@"
