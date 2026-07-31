#!/bin/bash

export PYTHONWARNINGS="ignore::UserWarning:pkg_resources"
# Define the dataset to evaluate
DATASET="epistr"
DATA_PATH="./data"
OUTPUT_PATH="./data/models/$DATASET"
# 1. Models designated for LINEAR PROBING
# Typically uses a higher learning rate and fewer epochs
LP_CLASSIFIERS=(
    "resnet18"
    "densenet121"
    "vit_base_patch16_224"
    "swin_base_patch4_window7_224"
    "ViT-B-16"
    "dinov2_vitb14"
)

# 2. Models designated for FULL FINETUNING
# Requires a much lower learning rate to preserve pre-trained weights
FT_CLASSIFIERS=(
    "resnet18"
    "densenet121"
    "vit_base_patch16_224"
)

echo "=== Starting Linear Probing Experiments ==="
for CL in "${LP_CLASSIFIERS[@]}"
do
    echo "Running LP for model: $CL"
    python -m experiments.train \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --classifier "$CL" \
        --augmentations "random_flip" "random_resized_crop" \
        --train_mode "linear_probe" \
        --epochs 50 \
        --lr 0.01 \
        --batch_size 32 \
        --seed 42 \
        --output_path "$OUTPUT_PATH" \
        --use_cuda 2>&1 | tee "logs/${DATASET}-${CL}-linear_probe.log"
done

echo "=== Starting Full Finetuning Experiments ==="
for CL in "${FT_CLASSIFIERS[@]}"
do
    echo "Running FT for model: $CL"
    python -m experiments.train \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --classifier "$CL" \
        --train_mode "finetune" \
        --augmentations "random_flip" "random_resized_crop" \
        --epochs 20 \
        --lr 0.0003 \
        --batch_size 32 \
        --seed 42 \
        --output_path "$OUTPUT_PATH" \
        --use_cuda 2>&1 | tee "logs/${DATASET}-${CL}-finetune.log"
done
