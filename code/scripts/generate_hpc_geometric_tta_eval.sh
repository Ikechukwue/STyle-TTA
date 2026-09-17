#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Geometric TTA Baseline (Production Edition)
# ============================================================================
# Sweeps: classifiers × eval_strategies × seeds on ImageNet-R
# Fully aligned with the high-performance local staging & cache frameworks.
#
# Usage:
#   bash scripts/generate_hpc_geometric_tta_eval.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TTA_METHOD="geometric"
AUGMENTATION="none"
DATASET="imagenet"
SPLIT="test_r_c26"

TARGET_DIR="${SCRIPT_DIR}/generated/geometric_tta"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "========================================================================"
echo "Geometric TTA Baseline Framework — HPC Script Generator"
echo "  Classifiers : ${ALL_CLASSIFIERS[*]}"
echo "  Eval strats : ${EVAL_STRATEGIES[*]}"
echo "  Seeds       : ${ALL_SEEDS[*]}"
echo "  Output Path : results/master_tta_results"
echo "========================================================================"

COUNTER=0

# Helper: generate one geometric TTA execution script
generate_script() {
    local CLF=$1
    local EVAL=$2
    local SEED=$3
    
    # Keeping the pretrained model guard bypass active 
    # (Matches your RetriStyle design requirement)
    # is_pretrained "$CLF" && return 0 
    
    local CLF_SAFE=$(safe_name "$CLF")

    # Allocation profiles for lightweight geometric transformations
    local TIER=$(echo "gpu:a100:1 a100")
    local GPU_CONFIG=$(echo "$TIER" | awk '{print $1}')
    local PARTITION=$(echo "$TIER" | awk '{print $2}')

    local WEIGHTS_LINE='WEIGHTS_PATH="pretrained"'
    if is_pretrained "$CLF"; then
        WEIGHTS_LINE='WEIGHTS_PATH=$WORK/retristyle/data/imagenet-$CLASSIFIER-random_flip-random_resized_crop-seed42.pth'
    fi

    local SCRIPT="${TARGET_DIR}/geo_${CLF_SAFE}_${EVAL}_s${SEED}.sh"

    cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=geo-${CLF_SAFE}-${EVAL}-s${SEED}
#SBATCH --gres=${GPU_CONFIG}
#SBATCH --partition=${PARTITION}
#SBATCH --time=1:30:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

TTA_METHOD="${TTA_METHOD}"
EVAL_STRATEGY="${EVAL}"
CLASSIFIER="${CLF}"
AUGMENTATION="${AUGMENTATION}"
DATASET="${DATASET}"
SEED=${SEED}
SPLIT="${SPLIT}"
${WEIGHTS_LINE}

CONTAINER=\$(eval echo ${HPC_CONTAINER})
DATA_PATH=\$(eval echo ${HPC_DATA_PATH})
OUTPUT_PATH=\$(eval echo ${HPC_OUTPUT_PATH})
EMBEDDING_DIR=\$(eval echo ${HPC_EMBEDDING_DIR})
HF_MODELS_CACHE=\$(eval echo ${HPC_HF_CACHE})
TORCH_MODELS_CACHE=\$(eval echo ${HPC_TORCH_CACHE})
LIVE_CODE=\$HPCVAULT/snapshots/retristyle

echo "Starting Job: Geometric TTA Baseline | \$CLASSIFIER | \$EVAL_STRATEGY | seed=\$SEED | \$(date)"
echo "Job context ID: \$SLURM_JOB_ID"
mkdir -p \$OUTPUT_PATH \$EMBEDDING_DIR

[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)

if [[ -n "\$TMPDIR" ]]; then
    echo "Unpacking Source ImageNet to Node Local SSD..."
    rsync -ahW --progress \$HPCVAULT/data/imagenet_test_r_c26.tar \$TMPDIR/
    rsync -ahW --progress \$HPCVAULT/data/imagenet_test_r.tar \$TMPDIR/
    tar -xf \$TMPDIR/imagenet_test_r.tar -C \$TMPDIR/
    tar -xf \$TMPDIR/imagenet_test_r_c26.tar -C \$TMPDIR/data/imagenet/imagenetr/
    EFFECTIVE_DATA_PATH=\$TMPDIR/data
    LOCAL_CACHE=\$TMPDIR/data/augmented_cache
    mkdir -p \$LOCAL_CACHE
else
    EFFECTIVE_DATA_PATH=\$DATA_PATH
    LOCAL_CACHE=\$TMPDIR/data/augmented_cache
fi

WEIGHTS_CLI=""
if [[ "\$WEIGHTS_PATH" != "pretrained" ]] && [[ -f "\$WEIGHTS_PATH" ]]; then
    WEIGHTS_CLI="--weights_path \$WEIGHTS_PATH"
else
    WEIGHTS_CLI="--weights_path pretrained"
fi

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 45m apptainer exec --nv \\
    --pwd /app \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$OUTPUT_PATH:/app/results \\
    --bind \$HOME/retristyle/data/imagenet/imagenet_subsets.json:/app/data/imagenet/imagenet_subsets.json \\
    --bind \$LOCAL_CACHE:/app/data/augmented_cache \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$EMBEDDING_DIR:/app/data/embeddings \\
    --bind \$LIVE_CODE:/app \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    accelerate launch --config_file \$ACCELERATE_CONFIG \\
        -m experiments.tta.run_inference \\
        --dataset \$DATASET --data_path /app/data --split \$SPLIT \\
        --classifier \$CLASSIFIER \$WEIGHTS_CLI \\
        --tta_method \$TTA_METHOD --eval_strategy \$EVAL_STRATEGY \\
        --n_views 16 \\
        --seed \$SEED \\
        --output_path /app/results/master_tta_results

EXIT_CODE=\$?
echo "Done: \$EXIT_CODE | \$(date)"
[[ \$EXIT_CODE -eq 124 ]] && sbatch "\${BASH_SOURCE[0]}"

exit \$EXIT_CODE
EOF
    chmod +x "$SCRIPT"
    COUNTER=$((COUNTER + 1))
}

# Clean grid generation loops
for CLF in "${ALL_CLASSIFIERS[@]}"; do
    for EVAL in "${EVAL_STRATEGIES[@]}"; do
        for SEED in "${ALL_SEEDS[@]}"; do
            generate_script "$CLF" "$EVAL" "$SEED"
        done
    done
done

# Orchestration utility generation
cat > "${TARGET_DIR}/submit_all.sh" << 'SUBMIT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== Launching Geometric TTA Baseline Matrix ==="
for s in "$SCRIPT_DIR"/geo_*.sh; do
    [[ -f "$s" ]] || continue
    echo "  sbatch $(basename $s)"
    sbatch "$s"
    sleep 0.5
done
SUBMIT
chmod +x "${TARGET_DIR}/submit_all.sh"

echo ""
echo "Generated $COUNTER baseline scripts in ${TARGET_DIR}/"
echo "Submit: bash ${TARGET_DIR}/submit_all.sh"
