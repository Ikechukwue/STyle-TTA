#!/bin/bash
# ============================================================================
# Evaluate Classifiers (local)
# ============================================================================
# Evaluates trained classifiers on train/val/test splits.
#
# Usage:
#   bash scripts/classifier_evaluation.sh --classifier resnet18 \
#       --augmentation none --seed 71397589
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CLASSIFIER="resnet18"
AUGMENTATION="none"
SEED=$DEFAULT_SEED
DATASET="imagenet"

while [[ $# -gt 0 ]]; do
    case $1 in
        --classifier) CLASSIFIER="$2"; shift 2 ;;
        --augmentation) AUGMENTATION="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

WEIGHTS_PATH="pretrained"
if ! is_pretrained "$CLASSIFIER"; then
    WEIGHTS_PATH="${MODEL_DIR}/${DATASET}-${CLASSIFIER}-${AUGMENTATION}-seed${SEED}.pth"
fi

echo "Evaluating: $CLASSIFIER | aug=$AUGMENTATION | seed=$SEED"

python -m experiments.classifier_evaluation \
    --dataset "$DATASET" --data_path "$DATA_PATH" \
    --classifier "$CLASSIFIER" --weights_path "$WEIGHTS_PATH" \
    --augmentation "$AUGMENTATION" --seed "$SEED" \
    --output_path "$OUTPUT_PATH/classifier_eval"
