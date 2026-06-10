#!/bin/bash
# ============================================================================
# Extract DINO Embeddings (local)
# ============================================================================
# Pre-computes DINOv2/DINOv3 embeddings for the training set of a dataset.
# Required before running retrieval-based TTA (dino strategy).
#
# Usage:
#   bash scripts/extract_embeddings.sh
#   bash scripts/extract_embeddings.sh --dataset imagenet --split train
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"


CNN_CLASSIFIERS=("densenet121")
VIT_CLASSIFIERS=("vit_base_patch16_224" "swin_base_patch4_window7_224")
FM_CLASSIFIERS=("dinov2_vitb14" "ViT-B-16")
TIMM_CLASSIFIERS=("${CNN_CLASSIFIERS[@]}" "${VIT_CLASSIFIERS[@]}" "${FM_CLASSIFIERS[@]}")
DATASET="imagenet"
SPLIT="test_r"
for EMBEDDING_MODEL in "${FM_CLASSIFIERS[@]}"; do

    echo "Extracting embeddings: $DATASET / $SPLIT / $EMBEDDING_MODEL"

    python -m experiments.tta.extract_embeddings \
        --dataset "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" "train@$SPLIT" "val@$SPLIT"\
        --output_dir "$EMBEDDING_DIR" \
        --model_name "$EMBEDDING_MODEL" \
        --input_size 224
done
