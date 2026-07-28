#!/bin/bash
# ============================================================================
# Run Domain Shift Analysis (local)
# ============================================================================
# Quantifies the domain gap between ImageNet-1k and its shifted variants
# using texture, colour, and feature-space metrics.
#
# Usage:
#   bash scripts/domain_shift_analysis.sh
#   bash scripts/domain_shift_analysis.sh --target_split test_a
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TARGET_SPLIT="test_r"
echo "Domain shift analysis: imagenet/val → imagenet/${TARGET_SPLIT}"
for i in {1..4}; do
python -m experiments.thesis.reporting.feature_space.domain_shift_feature \
    --embedding_dir ./data/embeddings \
    --dataset imagenet \
    --split_domain test_r \
    --split_train train@test_r \
    --split_val val@test_r \
    --k_style $i \
    --no_umap \
    --backbones resnet18 densenet121 vit_base_patch16_224 swin_base_patch4_window7_224 dinov2_vitb14 ViT-B-16
done
