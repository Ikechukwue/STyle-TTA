#!/bin/bash

# Kill the 'outdated' package noise and warnings
export OUTDATED_IGNORE=1
export PYTHONWARNINGS="ignore"

DATASETS=("imagenet")
SPLIT=("test_r")
MODELS=("ViT-B-16" "dinov2_vitb14")
TTA_METHOD=("geometric")
THESIS_N_REFS_SWEEP=(64)
THESIS_SEEDS=(71397589 133560673 265017005)
EVAL_STRATEGY=("tpt" "vanilla" "zero")


for TS in "${THESIS_SEEDS[@]}"; do
    for DS in "${SPLIT[@]}"; do
        for MDL in "${MODELS[@]}"; do
            for TTA in "${TTA_METHOD[@]}"; do
                for ES in "${EVAL_STRATEGY[@]}"; do
                    for NS in "${THESIS_N_REFS_SWEEP[@]}"; do
                        echo "----------------------------------------------------------"
                        echo "RUNNING: Model=$MDL on Dataset=$DS"
                        echo "----------------------------------------------------------"

                        python -m experiments.tta.run_inference \
                            --dataset "imagenet" \
                            --data_path "./data" \
                            --weights_path "./data/models/imagenet-$MDL-random_flip-random_resized_crop-seed42.pth" \
                            --classifier "$MDL" \
                            --tta_method "$TTA" \
                            --eval_strategy "$ES" \
                            --split "$DS" \
                            --batch_size 128 \
                            --num_workers 4 \
                            --seed "$TS" \
                            --n_views "$NS" \
                            --augmented_cache "./data/feature_cache"
                        echo "Finished $MDL on $DS"
                    done
                done
            done
        done
    done
done
