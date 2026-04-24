#!/bin/bash

# Load project variables and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

# Configuration for the smoke test
TEST_DURATION=120
LOG_DIR="./data/smoke_test_logs"
mkdir -p "$LOG_DIR"

# Subsets for testing
TEST_MODELS=("ViT-B-16" "resnet18")
TEST_METHODS=("retristyle") # Now includes styleid
TEST_EVALS=("zero")
MAIN_DIR="/home/stud/nemmler/retristyle"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

for MDL in "${ALL_CLASSIFIERS[@]}"; do
    for TTA in "${TEST_METHODS[@]}"; do
        
        # Determine if we need to loop through retrieval strategies
        # If the method is styleid, we test all retrievers. Otherwise, we just use "none"
        CURRENT_RETRIEVERS=("dino")
        if [[ "$TTA" == "retristyle" ]]; then
            CURRENT_RETRIEVERS=("${RETRIEVAL_STRATEGIES[@]}")
        fi

        for RET in "${CURRENT_RETRIEVERS[@]}"; do
            for ES in "${EVAL_STRATEGIES[@]}"; do
                
                # Construct log name including retrieval method
                SAFE_NAME=$(echo "${MDL}_${TTA}_${RET}_${ES}" | tr ' /' '_')
                LOG_FILE="$LOG_DIR/${SAFE_NAME}.log"

                # Handle weights via your helper function (which is the wrong way currently)
                if is_pretrained "$MDL"; then
                    continue
                    #W_PATH="./data/models/imagenet-$MDL-random_flip-random_resized_crop-seed42.pth"                    
                else
                    W_PATH="pretrained"
                fi

                echo -n "Model: $MDL | Method: $TTA | Ret: $RET | Eval: $ES ... "

                # EXECUTE
                python -m experiments.tta.run_inference \
                    --dataset "imagenet" \
                    --data_path "$DATA_PATH" \
                    --weights_path "$W_PATH" \
                    --classifier "$MDL" \
                    --tta_method "$TTA" \
                    --retrieval_strategy "$RET" \
                    --eval_strategy "$ES" \
                    --embedding_dir "$MAIN_DIR/data/embeddings/imagenet/vit_base_patch16_dinov3_lvd1689m/train@test_r.pt" \
                    --embedding_model "$EMBEDDING_MODEL" \
                    --split "test_r" \
                    --batch_size 32 \
                    --augmented_cache "$MAIN_DIR/data/trash/aug" \
                    --n_views 8 \
                    --n_refs 8 \
                    --seed "${ALL_SEEDS[0]}" \
                    --output_path "$MAIN_DIR/data/trash" > "$LOG_FILE" 2>&1 &
                
                PID=$!

                # Timeout logic
                SECONDS_WAITED=0
                while [ $SECONDS_WAITED -lt $TEST_DURATION ] && ps -p $PID > /dev/null; do
                    sleep 5
                    ((SECONDS_WAITED+=5))
                done

                if ps -p $PID > /dev/null; then
                    kill $PID
                    echo -e "[\e[32mPASS\e[0m]"
                else
                    wait $PID
                    echo -e "[\e[31mFAIL\e[0m] (Code $?)"
                fi
            done
        done
    done
done
