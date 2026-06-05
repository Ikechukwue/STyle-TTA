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

export OUTDATED_IGNORE=1
export PYTHONWARNINGS="ignore"

MODELS=(
    "ViT-B-16" 
    "dinov2_vitb14" 
    )
    #"ViT-B-16@Zero"
    #"resnet18" 
    #"densenet121" 
    #"swin_base_patch4_window7_224" 
    #"vit_base_patch16_224" )
AUGMENTATION="none"
SEED=$DEFAULT_SEED
DATASET="imagenet"

for MDL in "${MODELS[@]}"; do
    echo "----------------------------------------------------------"
    echo "RUNNING EVALUATION: Model=$MDL on Split=$SPLIT"
    echo "----------------------------------------------------------"

    # Exact weight resolving logic from your TTA script
    WEIGHTS_PATH="pretrained"
    if is_pretrained "$MDL"; then
        WEIGHTS_PATH="$MODEL_DIR/imagenet-$MDL-random_flip-random_resized_crop-seed42.pth"
    fi

    echo "Weights target resolving to: $WEIGHTS_PATH"

    python -m experiments.classifier_evaluation \
        --dataset "$DATASET" \
        --data_path "./data" \
        --classifier "$MDL" \
        --weights_path "$WEIGHTS_PATH" \
        --method "$AUGMENTATION" \
        --seed "$SEED" \
        --output_path "results/classifier_eval" 

    echo "Finished evaluating $MDL"
done
