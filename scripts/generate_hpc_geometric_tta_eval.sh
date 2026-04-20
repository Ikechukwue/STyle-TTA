#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Geometric TTA Baseline
# ============================================================================
# Sweeps: classifiers × eval_strategies × seeds
# GPU: 1× A40 per job (geometric TTA is lightweight)
#
# Usage:
#   bash scripts/generate_hpc_geometric_tta_eval.sh
#   bash scripts/generated/geometric_tta/submit_all.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TARGET_DIR="${SCRIPT_DIR}/generated/geometric_tta"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

TTA_METHOD="geometric"
AUGMENTATION="none"
DATASET="imagenet"
SPLIT="test_r"

echo "========================================================================"
echo "Geometric TTA Baseline — HPC Script Generator"
echo "  Classifiers : ${ALL_CLASSIFIERS[*]}"
echo "  Eval strats : ${EVAL_STRATEGIES[*]}"
echo "  Seeds       : ${ALL_SEEDS[*]}"
echo "========================================================================"

COUNTER=0

for CLF in "${ALL_CLASSIFIERS[@]}"; do
    CLF_SAFE=$(safe_name "$CLF")
    for EVAL in "${EVAL_STRATEGIES[@]}"; do
    for SEED in "${ALL_SEEDS[@]}"; do
        SCRIPT="${TARGET_DIR}/geo_${CLF_SAFE}_${EVAL}_s${SEED}.sh"

        WEIGHTS_LINE='WEIGHTS_PATH="pretrained"'
        if ! is_pretrained "$CLF"; then
            WEIGHTS_LINE='WEIGHTS_PATH=$WORK/retristyle/models/training/${DATASET}-${CLASSIFIER}-none-seed${SEED}.pth'
        fi

        cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=geo-${CLF_SAFE}-${EVAL}-s${SEED}
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00
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

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
OUTPUT_PATH=${HPC_OUTPUT_PATH}/geometric_tta
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "Geometric TTA: \$CLASSIFIER | \$EVAL_STRATEGY | seed=\$SEED | \$(date)"
mkdir -p \$OUTPUT_PATH
[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)
EOF
        # TMPDIR staging
        cat >> "$SCRIPT" << 'STAGE'

if [[ -n "$TMPDIR" ]]; then
    mkdir -p $TMPDIR/data/imagenet
    for sub in imagenet1k imagenet-r imagenet-a imagenet-sketch imagenetv2-matched-frequency-format-val; do
        [[ -d "$DATA_PATH/imagenet/$sub" ]] && rsync -a "$DATA_PATH/imagenet/$sub/" "$TMPDIR/data/imagenet/$sub/"
    done
    EFFECTIVE_DATA_PATH=$TMPDIR/data
else
    EFFECTIVE_DATA_PATH=$DATA_PATH
fi

WEIGHTS_CLI=""
if [[ "$WEIGHTS_PATH" != "pretrained" ]] && [[ -f "$WEIGHTS_PATH" ]]; then
    WEIGHTS_CLI="--weights_path $WEIGHTS_PATH"
else
    WEIGHTS_CLI="--weights_path pretrained"
fi
STAGE

        cat >> "$SCRIPT" << EOF

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 23h apptainer exec --nv \\
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
        --n_views ${DEFAULT_N_VIEWS} --seed \$SEED \\
        --output_path /app/results

EXIT_CODE=\$?
echo "Done: \$EXIT_CODE | \$(date)"
[[ \$EXIT_CODE -eq 124 ]] && sbatch "\${BASH_SOURCE[0]}"
exit \$EXIT_CODE
EOF
        chmod +x "$SCRIPT"
        COUNTER=$((COUNTER + 1))
    done
    done
done

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
