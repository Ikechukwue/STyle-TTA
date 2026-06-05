#!/bin/bash
# ============================================================================
# Run Geometric TTA Baseline (local)
# ============================================================================
# Evaluates geometric augmentation TTA (ZERO paper baseline) on a single
# classifier/eval_strategy/seed combination.
#
# Usage:
#   bash scripts/geometric_tta_eval.sh \
#       --classifier ViT-B-16 --eval_strategy zero --seed 71397589
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CLASSIFIER="ViT-B-16"
EVAL_STRATEGY="zero"
SEED=$DEFAULT_SEED
DATASET="imagenet"
SPLIT="test_r_c26"

while [[ $# -gt 0 ]]; do
    case $1 in
        --classifier) CLASSIFIER="$2"; shift 2 ;;
        --eval_strategy) EVAL_STRATEGY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

WEIGHTS_PATH="pretrained"
if ! is_pretrained "$CLASSIFIER"; then
    WEIGHTS_PATH="${MODEL_DIR}/${DATASET}-${CLASSIFIER}-none-seed${SEED}.pth"
fi

echo "Geometric TTA: $CLASSIFIER | $EVAL_STRATEGY | seed=$SEED"

accelerate launch --config_file "${PROJECT_ROOT}/configs/single_gpu_0.yaml" \
    -m experiments.tta.run_inference \
    --dataset "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" \
    --classifier "$CLASSIFIER" --weights_path "$WEIGHTS_PATH" \
    --tta_method geometric --eval_strategy "$EVAL_STRATEGY" \
    --n_views $DEFAULT_N_VIEWS --seed "$SEED" \
    --output_path "$OUTPUT_PATH/geometric_tta"
