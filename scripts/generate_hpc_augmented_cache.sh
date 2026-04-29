#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Augmented Image Cache
# ============================================================================
# Pre-generates stylized views on HPC so classifiers can be evaluated cheaply.
#
# Usage:
#   bash scripts/generate_hpc_augmented_cache.sh
#   bash scripts/generated/augmented_cache/submit_all.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TARGET_DIR="${SCRIPT_DIR}/generated/augmented_cache"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

#BEST_RETRIEVAL="${BEST_RETRIEVAL:-dino}"
BEST_RETRIEVAL="random"
DATASET="imagenet"
SPLIT="test_r"
N_REFS=16

echo "========================================================================"
echo "Augmented Cache — HPC Script Generator"
echo "  Retrieval : ${BEST_RETRIEVAL}"
echo "  n_refs    : ${N_REFS}"
echo "  Seeds     : ${ALL_SEEDS[*]}"
echo "========================================================================"

COUNTER=0

for SEED in "${ALL_SEEDS[@]}"; do
    TIER=$(gpu_tier_for_nrefs "$N_REFS")
    GPU_CONFIG=$(echo "$TIER" | awk '{print $1}')
    PARTITION=$(echo "$TIER" | awk '{print $2}')

    SCRIPT="${TARGET_DIR}/cache_${DATASET}_nr${N_REFS}_s${SEED}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=cache-${DATASET}-nr${N_REFS}-s${SEED}
#SBATCH --gres=${GPU_CONFIG}
#SBATCH --partition=${PARTITION}
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
EMBEDDING_DIR=${HPC_EMBEDDING_DIR}
FINAL_DEST=${HPC_OUTPUT_PATH}/augmented_cache
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "Augmented Cache: ${DATASET}/${SPLIT} | n_refs=${N_REFS} | seed=${SEED} | \$(date)"
mkdir -p \$FINAL_DEST \$EMBEDDING_DIR

[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)

if [[ -n "\$TMPDIR" ]]; then
    echo "Unpacking Source ImageNet to SSD..."
    tar -xf \$WORK/retristyle/retristyle_data.tar -C \$TMPDIR/
    EFFECTIVE_DATA_PATH=\$TMPDIR/data

    cp \$HOME/retristyle/data/imagenet/imagenet_subsets.json \$EFFECTIVE_DATA_PATH/imagenet/

    LOCAL_CACHE=\$TMPDIR/stylized_out
    mkdir -p \$LOCAL_CACHE
else
    EFFECTIVE_DATA_PATH=\$DATA_PATH
fi

ACCELERATE_CONFIG="/app/configs/gpu_02.yaml"

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 23h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$LOCAL_CACHE:/app/cache \\
    --bind \$HOME/retristyle/experiments/data:/app/experiments/data:ro \\
    --bind \$EMBEDDING_DIR:/app/embeddings \\
    --bind \$WORK/retristlye/acc_generate_augmented_images.py:/app/experiments/tta/generate_augmented_images.py \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    accelerate launch --config_file $ACCELERATE_CONFIG \\
        -m experiments.tta.generate_augmented_images \\
        --dataset ${DATASET} --data_path /app/data --split ${SPLIT} \\
        --tta_method retristyle \\
        --retrieval_strategy ${BEST_RETRIEVAL} \\
        --n_refs ${N_REFS} --n_views ${DEFAULT_N_VIEWS} \\
        --embedding_dir /app/embeddings --embedding_model ${EMBEDDING_MODEL} \\
        --cache_root /app/cache \\
        --seed ${SEED}

GEN_EXIT=\$?

if [ \$GEN_EXIT -eq 0 ] || [ \$GEN_EXIT -eq 124 ]; then
    echo "Archiving generated images to \$FINAL_DEST..."
    # Note: Using -c to create a new archive. Filename includes seed.
    tar -cf "\$FINAL_DEST/stylized_random_s${SEED}.tar" -C \$LOCAL_CACHE .
    echo "Archive complete: stylized_${BEST_RETRIEVAL}_n${N_REFS}_s${SEED}.tar"
fi

[[ \$GEN_EXIT -eq 124 ]] && sbatch "\$0"

exit \$GEN_EXIT
EOF
    chmod +x "$SCRIPT"
    COUNTER=$((COUNTER + 1))
done

cat > "${TARGET_DIR}/submit_all.sh" << 'SUBMIT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$SCRIPT_DIR"/cache_*.sh; do
    echo "sbatch $(basename $s)"
    sbatch "$s"
    sleep 1
done
SUBMIT
chmod +x "${TARGET_DIR}/submit_all.sh"

echo "Generated $COUNTER scripts in ${TARGET_DIR}/"
