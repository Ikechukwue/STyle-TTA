#!/bin/bash

CLASSIFIER=("ViT-B-16" "dinov2_vitb14")

for CL in "${CLASSIFIER[@]}"
do
    python -m experiments.train \
        --dataset imagenet \
        --data_path ./data \
        --classifier $CL \
        --epochs 200 \
        --lr 0.01 \
        --batch_size 256 \
        --seed 42 \
        --augmentations none \
        --output_path ./data/models \
        --use_cuda
done
