#!/bin/bash
# ============================================================================
# Run Style Transfer TTA Ablation (local)
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

ALL_CLASSIFIERS=("resnet50")
EVAL_STRATEGY="vanilla"
RETRIEVAL_STRATEGY="dino"
ALL_N_REFS=(16)
SEED=$DEFAULT_SEED

DATASET=("imagenet")
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

for DT in "${DATASET}"; do
    for CL in "${ALL_CLASSIFIERS[@]}"; do
        #AUGMENTATION="random_flip-random_resized_crop"
        #SPLIT="test"
        WEIGHTS_PATH="pretrained"
        #if is_pretrained "$CL"; then
        #if [[ $DT == "eurosat" ]]; then
        #    AUGMENTATION="random_flip-random_resized_crop"
        #    SPLIT="ucmerced"
        #fi
        #    WEIGHTS_PATH="${MODEL_DIR}/${DT}/${DT}-${CL}-${AUGMENTATION}-seed42.pth"
        #fi

        for N_REFS in "${ALL_N_REFS[@]}"; do
            #for EV in "${EVAL_STRATEGIES[@]}"; do
            echo "Ablation: $CL | retr=$RETRIEVAL_STRATEGY | eval=$EV | n_refs=$N_REFS"
            echo $WEIGHTS_PATH
            python -m code.experiments.tta.run_inference \
                --dataset "$DT" --data_path "$DATA_PATH" --split "$SPLIT" \
                --classifier "$CL" --weights_path "$WEIGHTS_PATH" \
                --tta_method style_tta --eval_strategy "$EVAL_STRATEGY" \
                --retrieval_strategy "$RETRIEVAL_STRATEGY" \
                --n_refs "$N_REFS" --n_views $N_REFS \
                --style_batch_size $STYLE_BATCH_SIZE \
                --embedding_model "$EMBEDDING_MODEL" \
                --embedding_dir "$EMBEDDING_DIR" \
                --seed "$SEED" \
                --augmented_cache "$AUG_DIR" \
                --output_path "$OUTPUT_PATH/ablation/style_tta"
        done
    done
done
