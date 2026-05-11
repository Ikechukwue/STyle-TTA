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

DATASET="imagenet"
SPLIT="test_abl"
N_REFS=16

TARGET_DIR="${SCRIPT_DIR}/generated/augmented_cache/${SPLIT}"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "========================================================================"
echo "Augmented Cache — HPC Script Generator"
echo "  Retrievals: ${RETRIEVAL_STRATEGIES[*]}"
echo "  n_refs    : ${N_REFS}"
echo "  Seeds     : ${ALL_SEEDS[*]}"
echo "========================================================================"

COUNTER=0

for RETRIEVAL in "${RETRIEVAL_STRATEGIES[@]}"; do
    for SEED in "${ALL_SEEDS[@]}"; do
        TIER=$(gpu_tier_for_nrefs "$N_REFS")
        GPU_CONFIG=$(echo "$TIER" | awk '{print $1}')
        PARTITION=$(echo "$TIER" | awk '{print $2}')

        SCRIPT="${TARGET_DIR}/cache_${RETRIEVAL}_${DATASET}_nr${N_REFS}_s${SEED}.sh"
        cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=cache-${RETRIEVAL}-${DATASET}-nr${N_REFS}-s${SEED}
#SBATCH --gres=${GPU_CONFIG}
#SBATCH --partition=${PARTITION}
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Proxy for NHR@FAU
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# Explicitly evaluate paths from common.sh
CONTAINER=$(eval echo ${HPC_CONTAINER})
DATA_PATH=$(eval echo ${HPC_DATA_PATH})
EMBEDDING_DIR=$(eval echo ${HPC_EMBEDDING_DIR})
FINAL_DEST=\$HPCVAULT/augmented_cache/${SPLIT}/"${RETRIEVAL}_${N_REFS}_${DATASET}_${SPLIT}_s${SEED}"
HF_MODELS_CACHE=$(eval echo ${HPC_HF_CACHE})
TORCH_MODELS_CACHE=$(eval echo ${HPC_TORCH_CACHE})
LIVE_CODE=\$HOME/retristyle/retristyle

echo "Starting Job: \$(date)"
mkdir -p "\$FINAL_DEST" "\$EMBEDDING_DIR"

if [[ -n "\$TMPDIR" ]]; then
    echo "Staging data to SSD..."
    # Note: Ensure this tar path is correct!
    tar -xf \$HPCVAULT/data/imagenet_test_abl.tar -C \$TMPDIR/
    EFFECTIVE_DATA_PATH=\$TMPDIR/data

    LOCAL_CACHE=\$TMPDIR/stylized_out
    mkdir -p \$LOCAL_CACHE

    # Restore logic
    if [ -d "\$FINAL_DEST" ] && [ "\$(ls -A "\$FINAL_DEST")" ]; then
        echo "Restoring previous tars..."
        find "\$FINAL_DEST" -name "*.tar" -exec tar -xf {} -C "\$LOCAL_CACHE" \;
    fi
else
    EFFECTIVE_DATA_PATH=\$DATA_PATH
    LOCAL_CACHE=\$FINAL_DEST/raw_files # Fallback if no TMPDIR
    mkdir -p \$LOCAL_CACHE
fi

# Run Apptainer
APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 22h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$LOCAL_CACHE:/app/cache \\
    --bind \$LIVE_CODE:/app \\
    --bind \$EMBEDDING_DIR:/app/data/embeddings \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    accelerate launch -m experiments.tta.acc_generate_augmented_images \\
        --dataset ${DATASET} --data_path /app/data --split ${SPLIT} \\
        --tta_method retristyle \\
        --retrieval_strategy ${RETRIEVAL} \\
        --n_refs ${N_REFS} --n_views ${DEFAULT_N_VIEWS} \\
        --embedding_dir /app/data/embeddings --embedding_model ${EMBEDDING_MODEL} \\
        --cache_root /app/cache \\
        --seed ${SEED}

GEN_EXIT=\$?

if [[ -n "\$TMPDIR" ]]; then
    echo "Bundling results..."
    cd "\$LOCAL_CACHE"
    tar -cf "\$TMPDIR/part_\$SLURM_JOB_ID.tar" .
    rsync -av --remove-source-files "\$TMPDIR/part_\$SLURM_JOB_ID.tar" "\$FINAL_DEST/"
fi

# Auto-resubmit if timed out
[[ \$GEN_EXIT -eq 124 ]] && sbatch "\$0"

exit \$GEN_EXIT

EOF
        chmod +x "$SCRIPT"
        COUNTER=$((COUNTER + 1))
    done
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
