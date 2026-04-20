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

DATASET="imagenet"
SPLIT="train"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Extracting embeddings: $DATASET / $SPLIT / $EMBEDDING_MODEL"

python -m experiments.tta.extract_embeddings \
    --dataset_name "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" \
    --output_dir "$EMBEDDING_DIR" \
    --model_name "$EMBEDDING_MODEL" \
    --input_size 224
