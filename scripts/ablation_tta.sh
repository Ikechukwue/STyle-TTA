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

CLASSIFIER="vit_base_patch16_dinov3_lvd1689m"
EVAL_STRATEGY="zero"
RETRIEVAL_STRATEGY="dino"
ALL_N_REFS=(2 4 8 16)
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

WEIGHTS_PATH="pretrained"
if is_pretrained "$CLASSIFIER"; then
    WEIGHTS_PATH="${MODEL_DIR}/${DATASET}-${CLASSIFIER}-random_flip-random_resized_crop-seed42.pth"
fi
for N_REFS in "${ALL_N_REFS[@]}"; do
echo "Ablation: $CLASSIFIER | retr=$RETRIEVAL_STRATEGY | eval=$EVAL_STRATEGY | n_refs=$N_REFS"
echo $WEIGHTS_PATH
python -m experiments.tta.run_inference \
    --dataset "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" \
    --classifier "$CLASSIFIER" --weights_path "$WEIGHTS_PATH" \
    --tta_method retristyle --eval_strategy "$EVAL_STRATEGY" \
    --retrieval_strategy "$RETRIEVAL_STRATEGY" \
    --n_refs "$N_REFS" --n_views $N_REFS \
    --style_batch_size $STYLE_BATCH_SIZE \
    --embedding_model "$EMBEDDING_MODEL" \
    --embedding_dir "$EMBEDDING_DIR" \
    --seed "$SEED" \
    --augmented_cache "$AUG_DIR" \
    --output_path "$OUTPUT_PATH/ablation/retristyle"
done
