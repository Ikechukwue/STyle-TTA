#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Reference Baseline Training
# ============================================================================
# Generates training scripts for:
#   A) Standard augmentation baselines   (experiments.train)
#   B) DomainBed regularisation methods  (experiments.reference_methods.train_domainbed)
#   C) Single Domain Generalisation      (experiments.reference_methods.train_sdg)
#
# Across: classifiers × seeds × {ImageNet datasets (+ others commented out)}
#
# Usage:
#   bash scripts/generate_hpc_train_reference_baselines.sh
#   bash scripts/generated/reference_baselines/submit_all.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BASE_OUTPUT="${SCRIPT_DIR}/generated/reference_baselines"
rm -rf "$BASE_OUTPUT"
mkdir -p "$BASE_OUTPUT"

# ── Active datasets (ImageNet first) ──
DATASETS=("imagenet")
# Uncomment to add more:
# DATASETS+=("pacs" "vlcs" "office_home" "domainnet" "terra_incognita")
# DATASETS+=("camelyon17wilds" "epistr" "fitzpatrick17k" "retina")

# Only trainable classifiers (CLIP/DINOv2 are pretrained — no training needed)
TRAINABLE_CLASSIFIERS=("${CNN_CLASSIFIERS[@]}" "${VIT_CLASSIFIERS[@]}")

echo "========================================================================"
echo "Reference Baselines — HPC Script Generator"
echo "  Datasets    : ${DATASETS[*]}"
echo "  Classifiers : ${TRAINABLE_CLASSIFIERS[*]}"
echo "  Seeds       : ${ALL_SEEDS[*]}"
echo "  Aug methods : ${TRAINING_AUGMENTATIONS[*]}"
echo "  DomainBed   : ${DOMAINBED_METHODS[*]}"
echo "  SDG         : ${SDG_METHODS[*]}"
echo "========================================================================"

COUNTER=0

# ── Helper: TMPDIR staging ──
tmpdir_staging() {
    local DATASET=$1
    cat << TMPEOF

if [[ -n "\$TMPDIR" ]]; then
    echo "Staging data to \$TMPDIR..."
    mkdir -p \$TMPDIR/data
    if [[ -d "\$DATA_PATH/${DATASET}" ]]; then
        rsync -a "\$DATA_PATH/${DATASET}/" "\$TMPDIR/data/${DATASET}/"
    elif [[ -d "\$DATA_PATH/imagenet" ]]; then
        mkdir -p \$TMPDIR/data/imagenet
        for sub in imagenet1k imagenet-r imagenet-a imagenet-sketch; do
            [[ -d "\$DATA_PATH/imagenet/\$sub" ]] && rsync -a "\$DATA_PATH/imagenet/\$sub/" "\$TMPDIR/data/imagenet/\$sub/"
        done
    fi
    EFFECTIVE_DATA_PATH=\$TMPDIR/data
else
    EFFECTIVE_DATA_PATH=\$DATA_PATH
fi
TMPEOF
}

# ============================================================================
# A) Augmentation Baselines
# ============================================================================
AUG_DIR="${BASE_OUTPUT}/augmentation"
mkdir -p "$AUG_DIR"

for DATASET in "${DATASETS[@]}"; do
for CLF in "${TRAINABLE_CLASSIFIERS[@]}"; do
    CLF_SAFE=$(safe_name "$CLF")
    for AUG in "${TRAINING_AUGMENTATIONS[@]}"; do
    for SEED in "${ALL_SEEDS[@]}"; do
        SCRIPT="${AUG_DIR}/train_${DATASET}_${CLF_SAFE}_${AUG}_s${SEED}.sh"
        cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=tr-${CLF_SAFE}-${AUG}-s${SEED}
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
OUTPUT_PATH=${HPC_MODEL_DIR}
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "Training: ${CLF} | ${AUG} | seed=${SEED} | \$(date)"
mkdir -p \$OUTPUT_PATH
[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)
EOF
        tmpdir_staging "$DATASET" >> "$SCRIPT"
        cat >> "$SCRIPT" << EOF

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 23h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$OUTPUT_PATH:/app/output \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    accelerate launch --config_file \$ACCELERATE_CONFIG \\
        -m experiments.train \\
        --dataset ${DATASET} --data_path /app/data \\
        --classifier ${CLF} --augmentation ${AUG} \\
        --seed ${SEED} --output_path /app/output

echo "Done: \$? | \$(date)"
EOF
        chmod +x "$SCRIPT"
        COUNTER=$((COUNTER + 1))
    done
    done
done
done

# ============================================================================
# B) DomainBed Methods
# ============================================================================
DB_DIR="${BASE_OUTPUT}/domainbed"
mkdir -p "$DB_DIR"

for DATASET in "${DATASETS[@]}"; do
for CLF in "${TRAINABLE_CLASSIFIERS[@]}"; do
    CLF_SAFE=$(safe_name "$CLF")
    for ALG in "${DOMAINBED_METHODS[@]}"; do
    ALG_LOWER=$(echo "$ALG" | tr '[:upper:]' '[:lower:]')
    for SEED in "${ALL_SEEDS[@]}"; do
        SCRIPT="${DB_DIR}/train_${DATASET}_${CLF_SAFE}_${ALG_LOWER}_s${SEED}.sh"
        cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=db-${CLF_SAFE}-${ALG_LOWER}-s${SEED}
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
OUTPUT_PATH=${HPC_MODEL_DIR}
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "DomainBed: ${CLF} | ${ALG} | seed=${SEED} | \$(date)"
mkdir -p \$OUTPUT_PATH
[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)
EOF
        tmpdir_staging "$DATASET" >> "$SCRIPT"
        cat >> "$SCRIPT" << EOF

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 23h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$OUTPUT_PATH:/app/output \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    accelerate launch --config_file \$ACCELERATE_CONFIG \\
        -m experiments.reference_methods.train_domainbed \\
        --dataset ${DATASET} --data_path /app/data \\
        --classifier ${CLF} --algorithm ${ALG} \\
        --seed ${SEED} --output_path /app/output

echo "Done: \$? | \$(date)"
EOF
        chmod +x "$SCRIPT"
        COUNTER=$((COUNTER + 1))
    done
    done
done
done

# ============================================================================
# C) SDG Methods
# ============================================================================
SDG_DIR="${BASE_OUTPUT}/sdg"
mkdir -p "$SDG_DIR"

for DATASET in "${DATASETS[@]}"; do
for CLF in "${TRAINABLE_CLASSIFIERS[@]}"; do
    CLF_SAFE=$(safe_name "$CLF")
    for METH in "${SDG_METHODS[@]}"; do
    METH_LOWER=$(echo "$METH" | tr '[:upper:]' '[:lower:]')
    for SEED in "${ALL_SEEDS[@]}"; do
        SCRIPT="${SDG_DIR}/train_${DATASET}_${CLF_SAFE}_${METH_LOWER}_s${SEED}.sh"
        cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=sdg-${CLF_SAFE}-${METH_LOWER}-s${SEED}
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
OUTPUT_PATH=${HPC_MODEL_DIR}
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "SDG: ${CLF} | ${METH} | seed=${SEED} | \$(date)"
mkdir -p \$OUTPUT_PATH
[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)
EOF
        tmpdir_staging "$DATASET" >> "$SCRIPT"
        cat >> "$SCRIPT" << EOF

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
timeout 23h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$OUTPUT_PATH:/app/output \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    accelerate launch --config_file \$ACCELERATE_CONFIG \\
        -m experiments.reference_methods.train_sdg \\
        --dataset ${DATASET} --data_path /app/data \\
        --classifier ${CLF} --sdg_method ${METH} \\
        --seed ${SEED} --output_path /app/output

echo "Done: \$? | \$(date)"
EOF
        chmod +x "$SCRIPT"
        COUNTER=$((COUNTER + 1))
    done
    done
done
done

# ============================================================================
# Submit-all
# ============================================================================
cat > "${BASE_OUTPUT}/submit_all.sh" << 'SUBMIT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dir in augmentation domainbed sdg; do
    [[ -d "$SCRIPT_DIR/$dir" ]] || continue
    echo "=== Submitting $dir ==="
    for s in "$SCRIPT_DIR/$dir"/train_*.sh; do
        echo "  sbatch $(basename $s)"
        sbatch "$s"
        sleep 0.5
    done
done
SUBMIT
chmod +x "${BASE_OUTPUT}/submit_all.sh"

echo ""
echo "Generated $COUNTER training scripts in ${BASE_OUTPUT}/"
echo "Submit: bash ${BASE_OUTPUT}/submit_all.sh"
