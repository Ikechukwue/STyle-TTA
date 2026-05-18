#!/bin/bash

CLASSIFIER=("ViT-B-16" "dinov2_vitb14")

for CL in "${CLASSIFIER[@]}"
do
    python -m experiments.train \
        --dataset imagenet \
        --data_path ./data \
        --classifier $CL \
        --epochs 100 \
        --lr 0.01 \
        --batch_size 256 \
        --seed 42 \
        --augmentations random_flip random_resized_crop \
        --output_path ./data/models \
        --use_cuda
done
