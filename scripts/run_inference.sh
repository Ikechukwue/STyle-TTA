#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

python -m experiments.tta.run_inference \
    --dataset imagenet --data_path ./data --split test_abl \
    --classifier resnet18 --weights_path pretrained \
    --tta_method adain_tta --eval_strategy vanilla \
    --retrieval_strategy random \
    --n_refs 32 --n_views 32 \
    --style_batch_size 8 \
    --embedding_model ${EMBEDDING_MODEL} \
    --augmented_cache ./data/augmented_cache \
    --embedding_dir ./data/embeddings \
    --seed ${DEFAULT_SEED} \
    --output_path ${OUTPUT_PATH}
