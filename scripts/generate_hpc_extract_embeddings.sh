#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Extract Embeddings
# ============================================================================
# Pre-computes DINO embeddings for all active datasets (training split).
#
# Usage:
#   bash scripts/generate_hpc_extract_embeddings.sh
#   bash scripts/generated/extract_embeddings/submit_all.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TARGET_DIR="${SCRIPT_DIR}/generated/extract_embeddings"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

DATASETS=("imagenet")
# Uncomment when extending:
# DATASETS+=("pacs" "vlcs" "office_home" "domainnet" "terra_incognita")
# DATASETS+=("camelyon17wilds" "epistr" "fitzpatrick17k" "retina")

echo "========================================================================"
echo "Embedding Extraction — HPC Script Generator"
echo "  Datasets: ${DATASETS[*]}"
echo "  Model   : ${EMBEDDING_MODEL}"
echo "========================================================================"

COUNTER=0

for DATASET in "${DATASETS[@]}"; do
    SCRIPT="${TARGET_DIR}/embed_${DATASET}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=emb-${DATASET}
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=12:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
EMBEDDING_DIR=${HPC_EMBEDDING_DIR}
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "Embedding extraction: ${DATASET} | \$(date)"
mkdir -p \$EMBEDDING_DIR

[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

APPTAINERENV_PYTHONPATH=/app \\
APPTAINERENV_http_proxy=\$http_proxy \\
APPTAINERENV_https_proxy=\$https_proxy \\
APPTAINERENV_HF_HOME=/app/hf_models \\
APPTAINERENV_TORCH_HOME=/app/torch_models \\
apptainer exec --nv \\
    --bind \$DATA_PATH:/app/data \\
    --bind \$EMBEDDING_DIR:/app/embeddings \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    python -m experiments.tta.extract_embeddings \\
        --dataset_name ${DATASET} --data_path /app/data --split train \\
        --output_dir /app/embeddings \\
        --model_name ${EMBEDDING_MODEL} \\
        --input_size 224

echo "Done: \$? | \$(date)"
EOF
    chmod +x "$SCRIPT"
    COUNTER=$((COUNTER + 1))
done

cat > "${TARGET_DIR}/submit_all.sh" << 'SUBMIT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$SCRIPT_DIR"/embed_*.sh; do
    echo "sbatch $(basename $s)"
    sbatch "$s"
done
SUBMIT
chmod +x "${TARGET_DIR}/submit_all.sh"

echo "Generated $COUNTER scripts in ${TARGET_DIR}/"
