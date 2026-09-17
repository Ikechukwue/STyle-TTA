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

DATASET=("camelyon17wilds")

SEED=$DEFAULT_SEED

for DS in "${DATASET[@]}"; do
    for MDL in "${MODELS[@]}"; do
        AUGMENTATION="random_flip-random_resized_crop"
        SPLITS=("train" "val" "test")
        if [[ "$DS" == "eurosat" ]]; then
            AUGMENTATION="random_flip-random_resized_crop"
            SPLITS=("train" "val" "train@ucmerced" "val@ucmerced" "ucmerced")
        fi
        echo "----------------------------------------------------------"
        echo "RUNNING EVALUATION: Model=$MDL on Dataset=$DS"
        echo "Splits: ${SPLITS[*]}"
        echo "----------------------------------------------------------"

        WEIGHTS_PATH="$MODEL_DIR/$DS/$DS-$MDL-$AUGMENTATION-seed42.pth"

        echo "Weights target resolving to: $WEIGHTS_PATH"

        # Pass the entire SPLITS array directly to the argument
        python -m experiments.classifier_evaluation \
            --dataset "$DS" \
            --data_path "./data" \
            --classifier "$MDL" \
            --weights_path "$WEIGHTS_PATH" \
            --method "$AUGMENTATION" \
            --seed "$SEED" \
            --splits "${SPLITS[@]}" \
            --output_path "results/classifier_eval"

        echo "Finished evaluating $MDL"
    done
done
