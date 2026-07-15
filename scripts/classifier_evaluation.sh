#!/bin/bash
# ============================================================================
# Evaluate Classifiers (local)
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export OUTDATED_IGNORE=1
export PYTHONWARNINGS="ignore"

MODELS=(
    "ViT-B-16"
    "dinov2_vitb14"
    "resnet18"
    "densenet121"
    "swin_base_patch4_window7_224"
    "vit_base_patch16_224"
)

# Define splits to evaluate all at once
SPLITS=("train" "val" "test")
DATASET="camelyon17wilds"
AUGMENTATION="none"
SEED=$DEFAULT_SEED

for MDL in "${MODELS[@]}"; do
    echo "----------------------------------------------------------"
    echo "RUNNING EVALUATION: Model=$MDL on Dataset=$DATASET"
    echo "Splits: ${SPLITS[*]}"
    echo "----------------------------------------------------------"

    WEIGHTS_PATH="$MODEL_DIR/$DATASET/$DATASET-$MDL-random_flip-random_resized_crop-seed42.pth"

    echo "Weights target resolving to: $WEIGHTS_PATH"

    # Pass the entire SPLITS array directly to the argument
    python -m experiments.classifier_evaluation \
        --dataset "$DATASET" \
        --data_path "./data" \
        --classifier "$MDL" \
        --weights_path "$WEIGHTS_PATH" \
        --method "$AUGMENTATION" \
        --seed "$SEED" \
        --splits "${SPLITS[@]}" \
        --output_path "results/classifier_eval"

    echo "Finished evaluating $MDL"
done
