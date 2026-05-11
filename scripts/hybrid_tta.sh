#!/bin/bash
# ============================================================================
# Run Hybrid TTA (local)
# ============================================================================
# Evaluates mixed geometric + style-transfer TTA at a given geo/style ratio.
#
# Usage:
#   bash scripts/hybrid_tta.sh --classifier ViT-B-16 --geo_frac 0.5
#   bash scripts/hybrid_tta.sh --classifier resnet18 --geo_frac 0.75 --seed 71397589
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CLASSIFIER="ViT-B-16"
GEO_FRAC=0.5
EVAL_STRATEGY="${BEST_EVAL:-zero}"
RETRIEVAL_STRATEGY="${BEST_RETRIEVAL:-dino}"
N_REFS="${BEST_N_REFS:-16}"
SEED=$DEFAULT_SEED
DATASET="imagenet"
SPLIT="test_r"

while [[ $# -gt 0 ]]; do
    case $1 in
        --classifier) CLASSIFIER="$2"; shift 2 ;;
        --geo_frac) GEO_FRAC="$2"; shift 2 ;;
        --eval_strategy) EVAL_STRATEGY="$2"; shift 2 ;;
        --retrieval_strategy) RETRIEVAL_STRATEGY="$2"; shift 2 ;;
        --n_refs) N_REFS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

WEIGHTS_PATH="pretrained"
if ! is_pretrained "$CLASSIFIER"; then
    WEIGHTS_PATH="${MODEL_DIR}/${DATASET}-${CLASSIFIER}-none-seed${SEED}.pth"
fi

echo "Hybrid TTA: $CLASSIFIER | geo=${GEO_FRAC} | eval=${EVAL_STRATEGY} | seed=${SEED}"

python -m experiments.tta.hybrid \
    --dataset "$DATASET" --split "$SPLIT" --data_path "$DATA_PATH" \
    --classifier "$CLASSIFIER" --weights_path "$WEIGHTS_PATH" \
    --geo_frac "$GEO_FRAC" \
    --n_views $DEFAULT_N_VIEWS --n_refs "$N_REFS" \
    --eval_strategy "$EVAL_STRATEGY" \
    --retrieval_strategy "$RETRIEVAL_STRATEGY" \
    --embedding_dir "$EMBEDDING_DIR" --embedding_model "$EMBEDDING_MODEL" \
    --seed "$SEED" \
    --output_path "$OUTPUT_PATH/hybrid_tta"
