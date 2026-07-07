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

# Array of tuples: "dataset:split"
DATASET_SPLITS=(
    "eurosat:train"
    "eurosat:val"
    "eurosat:val@ucmerced"
    "ucmerced:ucmerced"
)

AUGMENTATION="none"
SEED=$DEFAULT_SEED

for DS_SPLIT in "${DATASET_SPLITS[@]}"; do
    DATASET="${DS_SPLIT%%:*}"
    SPLIT="${DS_SPLIT#*:}"

    for MDL in "${MODELS[@]}"; do
        echo "----------------------------------------------------------"
        echo "RUNNING EVALUATION: Model=$MDL on Dataset=$DATASET Split=$SPLIT"
        echo "----------------------------------------------------------"

        # Adjust path logic as needed for your specific naming convention
        WEIGHTS_PATH="$MODEL_DIR/eurosat-$MDL-random_flip-random_resized_crop-seed42.pth"

        echo "Weights target resolving to: $WEIGHTS_PATH"

        python -m experiments.classifier_evaluation \
            --dataset "$DATASET" \
            --data_path "./data" \
            --classifier "$MDL" \
            --weights_path "$WEIGHTS_PATH" \
            --method "$AUGMENTATION" \
            --seed "$SEED" \
            --splits "$SPLIT" \
            --output_path "results/classifier_eval"

        echo "Finished evaluating $MDL"
    done
done
