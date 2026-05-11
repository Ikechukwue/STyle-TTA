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
SPLIT=("test_abl")
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
    SCRIPT="${TARGET_DIR}/embed_${DATASET}_${SPLIT}.sh"
    cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=emb-${DATASET}-${SPLIT}
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
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
if [[ -n "\$TMPDIR" ]]; then
    echo "Unpacking Source ImageNet to SSD..."
    tar -xf \$WORK/retristyle/data/${DATASET}_${SPLIT}.tar -C \$TMPDIR/
    DATA_PATH=\$TMPDIR/data

else
    DATA_PATH=${HPC_DATA_PATH}
fi
echo "Embedding extraction: ${DATASET} | ${SPLIT} |\$(date)"
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
    --bind \$HOME/retristyle/experiments/data:/app/experiments/data:ro \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$HOME/retristyle/reference_db.py:/app/retristyle/retrieval/reference_db.py \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    python -m experiments.tta.extract_embeddings \\
        --dataset ${DATASET} --data_path /app/data --split train@${SPLIT} ${SPLIT} \\
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
