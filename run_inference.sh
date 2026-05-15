#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"
# Kill the 'outdated' package noise and warnings
export OUTDATED_IGNORE=1
export PYTHONWARNINGS="ignore"

DATASETS=("imagenet")
SPLIT=("test_abl")
MODELS=("resnet18") #"densenet121" "swin_base_patch4_window7_224" "vit_base_patch16_224" ) #("ViT-B-16" "dinov2_vitb14") 
TTA_METHOD=("vanilla" ) #"zero" "tpt")
THESIS_N_REFS_SWEEP=(8 4 2 16)
THESIS_SEEDS=(133560673)
EVAL_STRATEGY=("random") # "dino" "balanced_random" "metric" "balanced_metric" 
MODEL_DIR="/home/stud/nemmler/retristyle/data/models"


for TS in "${THESIS_SEEDS[@]}"; do
    for DS in "${SPLIT[@]}"; do
        for MDL in "${MODELS[@]}"; do
            for TTA in "${TTA_METHOD[@]}"; do
                for ES in "${EVAL_STRATEGY[@]}"; do
                    for NS in "${THESIS_N_REFS_SWEEP[@]}"; do
                        echo "----------------------------------------------------------"
                        echo "RUNNING: Model=$MDL on Dataset=$DS"
                        echo "----------------------------------------------------------"

                        WEIGHTS_PATH="pretrained"
                        if is_pretrained "$MDL"; then
                            WEIGHTS_PATH="$MODEL_DIR/imagenet-$MDL-random_flip-random_resized_crop-seed42.pth"
                        fi

                        python -m experiments.tta.run_inference \
                            --dataset "imagenet" \
                            --data_path "./data" \
                            --weights_path $WEIGHTS_PATH \
                            --classifier $MDL \
                            --tta_method "retristyle" \
                            --eval_strategy $TTA \
                            --split $DS \
                            --retrieval_strategy $ES \
                            --n_refs $NS \
                            --batch_size 128 \
                            --num_workers 4 \
                            --seed "$TS" \
                            --n_views "$NS" \
                            --augmented_cache "./data/augmented_cache" \
                            --output_path "results/baseline/$DS"
                        echo "Finished $MDL on $DS"
                    done
                done
            done
        done
    done
done
