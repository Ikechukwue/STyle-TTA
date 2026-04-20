#!/bin/bash
# ============================================================================
# Train Reference Baselines (local)
# ============================================================================
# Trains a single reference baseline combination locally.
#
# Usage:
#   # Augmentation baseline
#   bash scripts/train_reference_baselines.sh --method augmentation \
#       --classifier resnet18 --augmentation color_jitter --seed 71397589
#
#   # DomainBed method
#   bash scripts/train_reference_baselines.sh --method domainbed \
#       --classifier resnet18 --domainbed_algorithm ERM --seed 71397589
#
#   # SDG method
#   bash scripts/train_reference_baselines.sh --method sdg \
#       --classifier resnet18 --sdg_method MixStyle --seed 71397589
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
METHOD="augmentation"
CLASSIFIER="resnet18"
AUGMENTATION="none"
DOMAINBED_ALG="ERM"
SDG_METHOD="MixStyle"
SEED=$DEFAULT_SEED
DATASET="imagenet"

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --classifier) CLASSIFIER="$2"; shift 2 ;;
        --augmentation) AUGMENTATION="$2"; shift 2 ;;
        --domainbed_algorithm) DOMAINBED_ALG="$2"; shift 2 ;;
        --sdg_method) SDG_METHOD="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

ACCELERATE_CONFIG="${PROJECT_ROOT}/configs/auto_gpu.yaml"

case $METHOD in
    augmentation)
        echo "Training: $CLASSIFIER | aug=$AUGMENTATION | seed=$SEED"
        accelerate launch --config_file "$ACCELERATE_CONFIG" \
            -m experiments.train \
            --dataset "$DATASET" --data_path "$DATA_PATH" \
            --classifier "$CLASSIFIER" --augmentation "$AUGMENTATION" \
            --seed "$SEED" --output_path "$OUTPUT_PATH/models/training"
        ;;
    domainbed)
        echo "Training DomainBed: $CLASSIFIER | $DOMAINBED_ALG | seed=$SEED"
        accelerate launch --config_file "$ACCELERATE_CONFIG" \
            -m experiments.reference_methods.train_domainbed \
            --dataset "$DATASET" --data_path "$DATA_PATH" \
            --classifier "$CLASSIFIER" --algorithm "$DOMAINBED_ALG" \
            --seed "$SEED" --output_path "$OUTPUT_PATH/models/training"
        ;;
    sdg)
        echo "Training SDG: $CLASSIFIER | $SDG_METHOD | seed=$SEED"
        accelerate launch --config_file "$ACCELERATE_CONFIG" \
            -m experiments.reference_methods.train_sdg \
            --dataset "$DATASET" --data_path "$DATA_PATH" \
            --classifier "$CLASSIFIER" --sdg_method "$SDG_METHOD" \
            --seed "$SEED" --output_path "$OUTPUT_PATH/models/training"
        ;;
    *)
        echo "Unknown method: $METHOD (use augmentation|domainbed|sdg)"
        exit 1 ;;
esac
