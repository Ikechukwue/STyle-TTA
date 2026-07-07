#!/bin/bash


# Define the dataset to evaluate
DATASET="eurosat"
DATA_PATH="./data"

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
LEFT=("ViT-B-16")
NOTHING=()
echo "=== Starting Linear Probing Experiments ==="
for CL in "${LEFT[@]}"
do
    echo "Running LP for model: $CL"
    python -m experiments.train \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --classifier "$CL" \
        --train_mode "linear_probe" \
        --epochs 50 \
        --lr 0.01 \
        --batch_size 256 \
        --seed 42 \
        --augmentations random_flip random_resized_crop \
        --output_path ./data/models \
        --use_cuda
done

echo "=== Starting Full Finetuning Experiments ==="
for CL in "${NOTHING[@]}"
do
    echo "Running FT for model: $CL"
    python -m experiments.train \
        --dataset "$DATASET" \
        --data_path "$DATA_PATH" \
        --classifier "$CL" \
        --train_mode "finetune" \
        --epochs 20 \
        --lr 0.0003 \
        --batch_size 128 \
        --seed 42 \
        --augmentations random_flip random_resized_crop \
        --output_path ./data/models \
        --use_cuda
done
