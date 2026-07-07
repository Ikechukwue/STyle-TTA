#!/bin/bash
# ============================================================================
# Generate Augmented Image Cache (local)
# ============================================================================
# Pre-generates stylized views for test images and saves to disk.
# Allows running multiple classifier evaluations without re-doing style transfer.
#
# Usage:
#   bash scripts/generate_augmented_cache.sh
#   bash scripts/generate_augmented_cache.sh --dataset imagenet --split test_r \
#       --n_refs 16 --retrieval_strategy dino
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

DATASET="eurosat"
SPLIT="ucmerced"
RETRIEVAL_STRATEGY="${BEST_RETRIEVAL:-dino}"
N_REFS=1
N_VIEWS=$DEFAULT_N_VIEWS
SEED=$DEFAULT_SEED

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --retrieval_strategy) RETRIEVAL_STRATEGY="$2"; shift 2 ;;
        --n_refs) N_REFS="$2"; shift 2 ;;
        --n_views) N_VIEWS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "Generating augmented cache: $DATASET/$SPLIT | retr=$RETRIEVAL_STRATEGY | n_refs=$N_REFS"

python -m experiments.tta.generate_augmented_images \
    --dataset "$DATASET" --data_path "$DATA_PATH" --split "$SPLIT" \
    --tta_method "retristyle"\
    --retrieval_strategy "$RETRIEVAL_STRATEGY" \
    --n_refs "$N_REFS" --n_views "$N_VIEWS" \
    --embedding_dir "$EMBEDDING_DIR" --embedding_model "$EMBEDDING_MODEL" \
    --cache_root "./data/augmented_cache" \
    --seed "$SEED"
