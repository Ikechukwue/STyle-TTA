#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Geometric TTA Baseline (Adapted)
# ============================================================================
# Sweeps: classifiers × eval_strategies × seeds
# Dynamic resource allocation & SSD optimization inherited from cache framework.
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
SPLIT="test_r"

TARGET_DIR="${SCRIPT_DIR}/generated/geometric_tta"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "========================================================================"
echo "Geometric TTA Baseline (Adapted) — HPC Script Generator"
echo "  Classifiers : ${ALL_CLASSIFIERS[*]}"
echo "  Eval strats : ${EVAL_STRATEGIES[*]}"
echo "  Seeds       : ${ALL_SEEDS[*]}"
echo "========================================================================"

generate_script() {
    local CLF=$1
    local EVAL=$2
    local SEED=$3

    local CLF_SAFE
    CLF_SAFE=$(safe_name "$CLF")

    # Dynamic resource checking based on your common.sh setups
    # (Defaults to low-tier resource configurations since Geometric TTA is lightweight)
    local TIER
    if type gpu_tier_for_nrefs &>/dev/null; then
        TIER=$(gpu_tier_for_nrefs "1") 
    else
        TIER="gpu:a40:1 a40"
    fi
    local GPU_CONFIG
    GPU_CONFIG=$(echo "$TIER" | awk '{print $1}')
    local PARTITION
    PARTITION=$(echo "$TIER" | awk '{print $2}')

    local SCRIPT="${TARGET_DIR}/geo_${CLF_SAFE}_${EVAL}_s${SEED}.sh"

    local WEIGHTS_LINE='WEIGHTS_PATH="pretrained"'
    if is_pretrained "$CLF"; then
        WEIGHTS_LINE='WEIGHTS_PATH=$WORK/retristyle/datag/${DATASET}-${CLASSIFIER}-random_flip-random_resized_crop-seed42.pth'
    fi

    cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=geo-${CLF_SAFE}-${EVAL}-s${SEED}
#SBATCH --gres=${GPU_CONFIG}
#SBATCH --partition=${PARTITION}
#SBATCH --time=1:05:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Proxy for NHR@FAU
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# Explicitly evaluate paths from common.sh configuration settings
CONTAINER=\$(eval echo ${HPC_CONTAINER})
DATA_PATH=\$(eval echo ${HPC_DATA_PATH})
OUTPUT_PATH=\$(eval echo ${HPC_OUTPUT_PATH})/geometric_tta
HF_MODELS_CACHE=\$(eval echo ${HPC_HF_CACHE})
TORCH_MODELS_CACHE=\$(eval echo ${HPC_TORCH_CACHE})
LIVE_CODE=\$(eval echo ${HPC_LIVE_CODE})

TTA_METHOD="${TTA_METHOD}"
EVAL_STRATEGY="${EVAL}"
CLASSIFIER="${CLF}"
AUGMENTATION="${AUGMENTATION}"
DATASET="${DATASET}"
SEED=${SEED}
SPLIT="${SPLIT}"
${WEIGHTS_LINE}

echo "Starting Job: Geometric TTA Baseline | \$CLASSIFIER | \$EVAL_STRATEGY | seed=\$SEED | \$(date)"
mkdir -p "\$OUTPUT_PATH"

if [[ -n "\$TMPDIR" ]]; then
    echo "Staging data using production tar methods..."
    # Adapts your high performance sequential staging framework
    rsync -ah --progress \$HPCVAULT/data/imagenet_test_r.tar "\$TMPDIR/tar_stage/"

    tar -xf "\$TMPDIR/tar_stage/imagenet_test_r.tar" -C \$TMPDIR/
    EFFECTIVE_DATA_PATH=\$TMPDIR/data
else
    EFFECTIVE_DATA_PATH=\$DATA_PATH
fi

WEIGHTS_CLI=""
if [[ "\$WEIGHTS_PATH" != "pretrained" ]] && [[ -f "\$WEIGHTS_PATH" ]]; then
    WEIGHTS_CLI="--weights_path \$WEIGHTS_PATH"
else
    WEIGHTS_CLI="--weights_path pretrained"
fi

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)

# Apptainer Execution Block
APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 1h apptainer exec --nv \\
    --pwd /app \\
    --bind \$LIVE_CODE:/app \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$OUTPUT_PATH:/app/results \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    accelerate launch --config_file \$ACCELERATE_CONFIG \\
        -m experiments.tta.run_inference \\
        --dataset \$DATASET --data_path /app/data --split \$SPLIT \\
        --classifier \$CLASSIFIER \$WEIGHTS_CLI \\
        --tta_method \$TTA_METHOD --eval_strategy \$EVAL_STRATEGY \\
        --n_views 16 --seed \$SEED \\
        --output_path /app/results

GEN_EXIT=\$?
echo "Done: \$GEN_EXIT | \$(date)"

# Auto-resubmit if timed out
[[ \$GEN_EXIT -eq 124 ]] && sbatch "\$0"

exit \$GEN_EXIT
EOF
    chmod +x "$SCRIPT"
    COUNTER=$((COUNTER + 1))
}

COUNTER=0

# Clean, encapsulated loop matrix using the generation block
for CLF in "${ALL_CLASSIFIERS[@]}"; do
    for EVAL in "${EVAL_STRATEGIES[@]}"; do
        for SEED in "${ALL_SEEDS[@]}"; do
            generate_script "$CLF" "$EVAL" "$SEED"
        done
    done
done

# Generate automated orchestration execution script
cat > "${TARGET_DIR}/submit_all.sh" << 'SUBMIT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$SCRIPT_DIR"/geo_*.sh; do
    echo "sbatch $(basename $s)"
    sbatch "$s"
    sleep 0.5
done
SUBMIT
chmod +x "${TARGET_DIR}/submit_all.sh"

echo "Generated $COUNTER scripts in ${TARGET_DIR}/"
echo "Submit: bash ${TARGET_DIR}/submit_all.sh"
