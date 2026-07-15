#!/bin/bash
# ============================================================================
# Run Hybrid TTA Loop (local)
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Define the list of classifiers to iterate over

EVAL_STRATEGY="${BEST_EVAL:-vanilla}"
RETRIEVAL_STRATEGY="${BEST_RETRIEVAL:-dino}"
N_REFS="${BEST_N_REFS:-1}"
N_VIEWS=(3 7 15 31 63)
SEED=$DEFAULT_SEED
DATASET="eurosat"
SPLIT="ucmerced"

while [[ $# -gt 0 ]]; do
    case $1 in
        --eval_strategy) EVAL_STRATEGY="$2"; shift 2 ;;
        --retrieval_strategy) RETRIEVAL_STRATEGY="$2"; shift 2 ;;
        --n_refs) N_REFS="$2"; shift 2 ;;
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

    # Inner loop: Views from 2 up to 63
    for N_VIEWS in "${N_VIEWS[@]}"; do
        # Calculate geo_frac: (N_VIEWS - 1) / N_VIEWS
        GEO_FRAC=$(awk "BEGIN {print ($N_VIEWS - 1) / $N_VIEWS}")

        echo "--------------------------------------------------------"
        echo "Hybrid TTA: $CLASSIFIER | views=$N_VIEWS | geo=${GEO_FRAC} | eval=${EVAL_STRATEGY}"
        echo "--------------------------------------------------------"

        python -m experiments.tta.hybrid \
            --dataset "$DATASET" --split "$SPLIT" --data_path "$DATA_PATH" \
            --classifier "$CLASSIFIER" --weights_path "$WEIGHTS_PATH" \
            --geo_frac "$GEO_FRAC" \
            --n_views "$N_VIEWS" --n_refs "$N_REFS" \
            --augmented_cache "./data/augmented_cache" \
            --eval_strategy "$EVAL_STRATEGY" \
            --retrieval_strategy "$RETRIEVAL_STRATEGY" \
            --embedding_dir "$EMBEDDING_DIR" --embedding_model "$EMBEDDING_MODEL" \
            --seed "$SEED" \
            --output_path "$OUTPUT_PATH/hybrid_tta/tta_inference"
    done
done
