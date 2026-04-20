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

BEST_RETRIEVAL="${BEST_RETRIEVAL:-dino}"
DATASET="imagenet"
SPLIT="test_r"
N_REFS="${BEST_N_REFS:-16}"

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
#SBATCH --time=48:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
EMBEDDING_DIR=${HPC_EMBEDDING_DIR}
CACHE_ROOT=${HPC_OUTPUT_PATH}/augmented_cache
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "Augmented Cache: ${DATASET}/${SPLIT} | n_refs=${N_REFS} | seed=${SEED} | \$(date)"
mkdir -p \$CACHE_ROOT \$EMBEDDING_DIR

[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)

if [[ -n "\$TMPDIR" ]]; then
    mkdir -p \$TMPDIR/data/imagenet
    for sub in imagenet1k imagenet-r; do
        [[ -d "\$DATA_PATH/imagenet/\$sub" ]] && rsync -a "\$DATA_PATH/imagenet/\$sub/" "\$TMPDIR/data/imagenet/\$sub/"
    done
    EFFECTIVE_DATA_PATH=\$TMPDIR/data
else
    EFFECTIVE_DATA_PATH=\$DATA_PATH
fi

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 47h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$CACHE_ROOT:/app/cache \\
    --bind \$EMBEDDING_DIR:/app/embeddings \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    python -m experiments.tta.generate_augmented_images \\
        --dataset ${DATASET} --data_path /app/data --split ${SPLIT} \\
        --tta_method retristyle \\
        --retrieval_strategy ${BEST_RETRIEVAL} \\
        --n_refs ${N_REFS} --n_views ${DEFAULT_N_VIEWS} \\
        --embedding_dir /app/embeddings --embedding_model ${EMBEDDING_MODEL} \\
        --cache_root /app/cache \\
        --seed ${SEED}

EXIT_CODE=\$?
echo "Done: \$EXIT_CODE | \$(date)"
[[ \$EXIT_CODE -eq 124 ]] && sbatch "\${BASH_SOURCE[0]}"
exit \$EXIT_CODE
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
