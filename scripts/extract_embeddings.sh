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

REST_CLASSIFIERS=("resnet18")
CNN_CLASSIFIERS=("densenet121" "resnet18")
VIT_CLASSIFIERS=("vit_base_patch16_224" "swin_base_patch4_window7_224")
FM_CLASSIFIERS=("ViT-B-16" "dinov2_vitb14" "vit_base_patch16_dinov3_lvd1689m")
TIMM_CLASSIFIERS=("${CNN_CLASSIFIERS[@]}" "${VIT_CLASSIFIERS[@]}" "${FM_CLASSIFIERS[@]}")
DATASET="imagenet"
SPLIT="test_r"
K_INTERVALS=(1 2 4 8 16)
for EMBEDDING_MODEL in "${REST_CLASSIFIERS[@]}"; do

    echo "Extracting embeddings: $DATASET / $SPLIT / $EMBEDDING_MODEL"

    python -m experiments.tta.extract_embeddings \
        --dataset "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" \
        --output_dir "$EMBEDDING_DIR" \
        --model_name "$EMBEDDING_MODEL" \
        --input_size 224 \
        --use_stylized "${K_INTERVALS[@]}"
done
