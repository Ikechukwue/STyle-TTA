#!/bin/bash
# ============================================================================
# Run Style Transfer TTA Image Prediciton (local)
# ============================================================================
# Runs a single ablation configuration for RetriStyle TTA.
#
# Usage:
#   # Retrieval strategy ablation
#   bash scripts/ablation_tta.sh --retrieval_strategy dino --eval_strategy zero \
#       --n_refs 16 --classifier ViT-B-16
#
#   # Eval strategy ablation
#   bash scripts/ablation_tta.sh --retrieval_strategy dino --eval_strategy tpt \
#       --n_refs 16 --classifier ViT-B-16
#
#   # n_refs sweep
#   bash scripts/ablation_tta.sh --retrieval_strategy dino --eval_strategy zero \
#       --n_refs 4 --classifier ViT-B-16
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"


EVAL_STRATEGY="vanilla"
RETRIEVAL_STRATEGY="dino"
SEED=$DEFAULT_SEED

DATASETS=("camelyon17wilds" "epistr" "eurosat")

for DT in "${DATASETS[@]}"; do
    SPLIT="test"
    AUGMENTATION="random_flip-random_resized_crop"

    if [[ $DT == "eurosat" ]]; then
        SPLIT="ucmerced"
    elif [[ $DT == "midog" ]]; then
        AUGMENTATION="none"
    fi

    for i in {1..4}; do
        name_id=$(printf "view_%03d" "$i")
        for CL in "${ALL_CLASSIFIERS[@]}"; do

            WEIGHTS_PATH="${MODEL_DIR}/${DT}/${DT}-${CL}-${AUGMENTATION}-seed42.pth"

            if [[ $DT == "imagenet" ]]; then
                is_pretrained=0
                for p_cls in "${PRETRAINED_CLASSIFIERS[@]}"; do
                    if [[ "$p_cls" == "$CL" ]]; then
                        is_pretrained=1
                        break
                    fi
                done
                if [[ $is_pretrained -eq 0 ]]; then
                    WEIGHTS_PATH="pretrained"
                fi
            fi

            echo "Test: $CL | Dataset: $DT | retr=$RETRIEVAL_STRATEGY | eval=$EVAL_STRATEGY | Style Image=$i"
            echo "$WEIGHTS_PATH"
            python -m code.experiments.tta.run_inference \
                --dataset "$DT" --data_path "$DATA_PATH" --split "$SPLIT" \
                --classifier "$CL" --weights_path "$WEIGHTS_PATH" \
                --tta_method geometric --eval_strategy "$EVAL_STRATEGY" \
                --style_id "$i" \
                --seed "$SEED" \
                --output_path "$OUTPUT_PATH/style_check/styleid/$name_id"
        done
    done
done
