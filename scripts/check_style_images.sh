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

DATASET="imagenet"
SPLIT="test_r"

while [[ $# -gt 0 ]]; do
    case $1 in
        --classifier) CLASSIFIER="$2"; shift 2 ;;
        --eval_strategy) EVAL_STRATEGY="$2"; shift 2 ;;
        --retrieval_strategy) RETRIEVAL_STRATEGY="$2"; shift 2 ;;
        --n_refs) N_REFS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done
for i in {1..15}; do
    name_id=$(printf "view_%03d.png" "$i")
    for CL in "${ALL_CLASSIFIERS[@]}"; do
        WEIGHTS_PATH="pretrained"
        if is_pretrained "$CL"; then
            WEIGHTS_PATH="${MODEL_DIR}/${DATASET}-${CL}-random_flip-random_resized_crop-seed42.pth"
        fi

        echo "Test: $CL | retr=$RETRIEVAL_STRATEGY | eval=$EV | Style Image=$i"
        echo $WEIGHTS_PATH
        python -m experiments.tta.run_inference \
            --dataset "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" \
            --classifier "$CL" --weights_path "$WEIGHTS_PATH" \
            --tta_method geometric --eval_strategy "$EVAL_STRATEGY" \
            --style_id "$i" \
            --seed "$SEED" \
            --output_path "$OUTPUT_PATH/style_check/adain/$name_id"
        #done
    done
done
