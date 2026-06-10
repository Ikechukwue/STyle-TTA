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

CNN_CLASSIFIERS=("densenet121")
VIT_CLASSIFIERS=("vit_base_patch16_224" "swin_base_patch4_window7_224")
FM_CLASSIFIERS=("dinov2_vitb14" "ViT-B-16")
MODELS=("${CNN_CLASSIFIERS[@]}" "${VIT_CLASSIFIERS[@]}" "${FM_CLASSIFIERS[@]}")
EVAL_STRATEGY="vanilla"
SEED=(71397589 133560673 265017005)
VIEWS=(1 32)
DATASET="imagenet"
SPLIT="val@test_r"

for MDL in "${MODELS[@]}"; do
    for VW in "${VIEWS[@]}"; do
        for SD in "${SEED[@]}"; do
            WEIGHTS_PATH="pretrained"
            
            if is_pretrained "$MDL"; then
                WEIGHTS_PATH="$MODEL_DIR/imagenet-$MDL-random_flip-random_resized_crop-seed42.pth"
            fi

            # 1. FIXED: Now logging the actual loop variable ($MDL) 
            echo "Geometric TTA: $MDL | $EVAL_STRATEGY | seed=$SD"

            accelerate launch --config_file "${PROJECT_ROOT}/configs/single_gpu_0.yaml" \
                -m experiments.tta.run_inference \
                --dataset "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" \
                --classifier "$MDL" --weights_path "$WEIGHTS_PATH" \
                --tta_method geometric --eval_strategy "$EVAL_STRATEGY" \
                --n_views "$VW" \
                --seed "$SD" \
                --output_path "$OUTPUT_PATH/geometric_tta"
                # 2. FIXED: Passed "$SEED" instead of $DEFAULT_SEED, and added quotes.
        done
    done
done
