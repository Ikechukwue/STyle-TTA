#!/bin/bash
# ============================================================================
# Run Hybrid TTA Loop (local)
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

EVAL_STRATEGY="${BEST_EVAL:-vanilla}"
RETRIEVAL_STRATEGY="${BEST_RETRIEVAL:-dino}"
ALL_SEEDS=(133560673 265017005)
SEED=$DEFAULT_SEED
DATASET=("imagenet" "midog" "camelyon17wilds" "epistr" "eurosat")
SPLIT="ucmerced"
N_VIEWS_LIST=(31 63)
N_REFS_LIST=(3)

for SD in "${ALL_SEEDS[@]}"; do 
    for DT in "${DATASET[@]}"; do
        for CLASSIFIER in "${ALL_CLASSIFIERS[@]}"; do

            AUGMENTATION="random_flip-random_resized_crop"
            SPLIT="test"

            if [[ $DT == "eurosat" ]]; then
                SPLIT="ucmerced"
            fi

            if [[ $DT == "midog" ]]; then
                AUGMENTATION="none"
            fi
            
            WEIGHTS_PATH="${MODEL_DIR}/${DT}/${DT}-${CLASSIFIER}-${AUGMENTATION}-seed42.pth"

            if [[ $DT == "imagenet" ]]; then
                SPLIT="test_r"
                is_pretrained=0
                for p_cls in "${PRETRAINED_CLASSIFIERS[@]}"; do
                    if [[ "$p_cls" == "$CLASSIFIER" ]]; then
                        is_pretrained=1
                        break
                    fi
                done
                if [[ $is_pretrained -eq 0 ]]; then
                    WEIGHTS_PATH="pretrained"
                fi
            fi
            
            for N_VIEWS in "${N_VIEWS_LIST[@]}"; do
                for N_REFS in "${N_REFS_LIST[@]}"; do

                    # --------------------------------------------------------
                    # Common quantities
                    # --------------------------------------------------------
                    N_GEO=$((N_VIEWS - N_REFS))

                    if [ "$N_REFS" -gt "$N_VIEWS" ]; then
                        continue
                    fi

                    # ========================================================
                    # RUN 1:
                    # Style references + original + remaining geometric
                    # views ALL generated from original
                    # ========================================================

                    echo "--------------------------------------------------------"
                    echo "Hybrid TTA [ORIGINAL GEO]"
                    echo "Classifier : $CLASSIFIER"
                    echo "Views      : $N_VIEWS"
                    echo "Style refs : $N_REFS"
                    echo "Geo views  : $N_GEO"
                    echo "Geo refs   : original only"
                    echo "--------------------------------------------------------"

                    python -m experiments.tta.hybrid \
                        --dataset "$DT" \
                        --split "$SPLIT" \
                        --data_path "$DATA_PATH" \
                        --classifier "$CLASSIFIER" \
                        --weights_path "$WEIGHTS_PATH" \
                        --n_views "$N_VIEWS" \
                        --n_refs "$N_REFS" \
                        --augmented_cache "./data/augmented_cache" \
                        --eval_strategy "$EVAL_STRATEGY" \
                        --retrieval_strategy "$RETRIEVAL_STRATEGY" \
                        --embedding_dir "$EMBEDDING_DIR" \
                        --embedding_model "$EMBEDDING_MODEL" \
                        --seed "$SD" \
                        --use_n_refs 1 \
                        --output_path "$OUTPUT_PATH/hybrid_tta/tta_inference"


                    # ========================================================
                    # RUN 2:
                    # Style references + original, with geometric views
                    # distributed across all references
                    # ========================================================

                    USE_N_REFS=$((N_REFS + 1))

                    echo "--------------------------------------------------------"
                    echo "Hybrid TTA [ALL REFS GEO]"
                    echo "Classifier : $CLASSIFIER"
                    echo "Views      : $N_VIEWS"
                    echo "Style refs : $N_REFS"
                    echo "Geo views  : $N_GEO"
                    echo "Geo refs   : original + $N_REFS style refs"
                    echo "--------------------------------------------------------"

                    python -m experiments.tta.hybrid \
                        --dataset "$DT" \
                        --split "$SPLIT" \
                        --data_path "$DATA_PATH" \
                        --classifier "$CLASSIFIER" \
                        --weights_path "$WEIGHTS_PATH" \
                        --n_views "$N_VIEWS" \
                        --n_refs "$N_REFS" \
                        --augmented_cache "./data/augmented_cache" \
                        --eval_strategy "$EVAL_STRATEGY" \
                        --retrieval_strategy "$RETRIEVAL_STRATEGY" \
                        --embedding_dir "$EMBEDDING_DIR" \
                        --embedding_model "$EMBEDDING_MODEL" \
                        --seed "$SD" \
                        --use_n_refs "$USE_N_REFS" \
                        --output_path "$OUTPUT_PATH/hybrid_tta/tta_inference"

                done
            done
        done
    done
done
