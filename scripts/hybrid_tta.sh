#!/bin/bash
# ============================================================================
# Run Hybrid TTA Loop (local)
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

EVAL_STRATEGY="${BEST_EVAL:-vanilla}"
RETRIEVAL_STRATEGY="${BEST_RETRIEVAL:-dino}"
N_VIEWS_LIST=(3 7 15 31 63)
N_REFS_LIST=(1)
SEED=$DEFAULT_SEED
DATASET="eurosat"
SPLIT="ucmerced"

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval_strategy) EVAL_STRATEGY="$2"; shift 2 ;;
        --retrieval_strategy) RETRIEVAL_STRATEGY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Outer loop: Classifiers
for CLASSIFIER in "${ALL_CLASSIFIERS[@]}"; do

    #WEIGHTS_PATH="pretrained"
    #if is_pretrained "$CLASSIFIER"; then
    WEIGHTS_PATH="${MODEL_DIR}/${DATASET}-${CLASSIFIER}-random_flip-random_resized_crop-seed42.pth"
    #fi

    for N_VIEWS in "${N_VIEWS_LIST[@]}"; do
        for N_REFS in "${N_REFS_LIST[@]}"; do
            
            # Skip configurations where stylized references exceed the total requested views
            if [ "$N_REFS" -gt "$N_VIEWS" ]; then
                continue
            fi

            N_GEO=$((N_VIEWS - N_REFS))

            echo "--------------------------------------------------------"
            echo "Hybrid TTA: $CLASSIFIER | views=$N_VIEWS | geo=$N_GEO | sty=$N_REFS | eval=${EVAL_STRATEGY}"
            echo "--------------------------------------------------------"

            python -m experiments.tta.hybrid \
                --dataset "$DATASET" --split "$SPLIT" --data_path "$DATA_PATH" \
                --classifier "$CLASSIFIER" --weights_path "$WEIGHTS_PATH" \
                --n_views "$N_VIEWS" --n_refs "$N_REFS" \
                --augmented_cache "./data/augmented_cache" \
                --eval_strategy "$EVAL_STRATEGY" \
                --retrieval_strategy "$RETRIEVAL_STRATEGY" \
                --embedding_dir "$EMBEDDING_DIR" --embedding_model "$EMBEDDING_MODEL" \
                --seed "$SEED" \
                --output_path "$OUTPUT_PATH/hybrid_tta/tta_inference"
        done
    done
done
