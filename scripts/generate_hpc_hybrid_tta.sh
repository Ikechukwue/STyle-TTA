#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Hybrid TTA (geo/style mixing)
# ============================================================================
# Sweeps geo/style ratios: 100/0, 75/25, 50/50, 25/75, 0/100
# Across all classifiers × seeds.
#
# Override best settings:
#   BEST_RETRIEVAL=dino BEST_EVAL=zero BEST_N_REFS=16 \
#       bash scripts/generate_hpc_hybrid_tta.sh
#
# Usage:
#   bash scripts/generate_hpc_hybrid_tta.sh
#   bash scripts/generated/hybrid_tta/submit_all.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BEST_RETRIEVAL="${BEST_RETRIEVAL:-dino}"
BEST_EVAL="${BEST_EVAL:-zero}"
BEST_N_REFS="${BEST_N_REFS:-16}"

TARGET_DIR="${SCRIPT_DIR}/generated/hybrid_tta"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

DATASET="imagenet"
SPLIT="test_r"

echo "========================================================================"
echo "Hybrid TTA — HPC Script Generator"
echo "  Geo fracs   : ${GEO_FRACS[*]}"
echo "  Classifiers : ${ALL_CLASSIFIERS[*]}"
echo "  Seeds       : ${ALL_SEEDS[*]}"
echo "  Best eval   : ${BEST_EVAL}"
echo "  Best retr   : ${BEST_RETRIEVAL}"
echo "  Best n_refs : ${BEST_N_REFS}"
echo "========================================================================"

COUNTER=0

for CLF in "${ALL_CLASSIFIERS[@]}"; do
    CLF_SAFE=$(safe_name "$CLF")
    for GEO in "${GEO_FRACS[@]}"; do
        GEO_TAG=$(echo "$GEO" | tr '.' 'p')
        for SEED in "${ALL_SEEDS[@]}"; do
            SCRIPT="${TARGET_DIR}/hybrid_${CLF_SAFE}_g${GEO_TAG}_s${SEED}.sh"

            WEIGHTS_LINE='WEIGHTS_PATH="pretrained"'
            if ! is_pretrained "$CLF"; then
                WEIGHTS_LINE='WEIGHTS_PATH=$WORK/retristyle/models/training/${DATASET}-${CLASSIFIER}-none-seed${SEED}.pth'
            fi

            # Style requires more GPU for small geo_frac
            GPU_CONFIG="gpu:a40:1"; PARTITION="a40"
            if [[ "$GEO" == "0.0" || "$GEO" == "0.25" ]]; then
                GPU_CONFIG="gpu:a40:2"; PARTITION="a40"
            fi

            cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=hyb-${CLF_SAFE}-g${GEO_TAG}-s${SEED}
#SBATCH --gres=${GPU_CONFIG}
#SBATCH --partition=${PARTITION}
#SBATCH --time=24:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

DATASET="${DATASET}"
CLASSIFIER="${CLF}"
GEO_FRAC=${GEO}
SEED=${SEED}
${WEIGHTS_LINE}

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
OUTPUT_PATH=${HPC_OUTPUT_PATH}/hybrid_tta
EMBEDDING_DIR=${HPC_EMBEDDING_DIR}
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}

echo "Hybrid TTA: geo=\$GEO_FRAC | \$CLASSIFIER | seed=\$SEED | \$(date)"
mkdir -p \$OUTPUT_PATH \$EMBEDDING_DIR

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
timeout 23h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$OUTPUT_PATH:/app/results \\
    --bind \$EMBEDDING_DIR:/app/embeddings \\
    --bind \$HF_MODELS_CACHE:/app/hf_models \\
    --bind \$TORCH_MODELS_CACHE:/app/torch_models \\
    \$CONTAINER \\
    python -m experiments.tta.hybrid \\
        --dataset ${DATASET} --split ${SPLIT} \\
        --data_path /app/data \\
        --classifier "\$CLASSIFIER" \\
        \$WEIGHTS_CLI \\
        --geo_frac \$GEO_FRAC \\
        --n_views ${DEFAULT_N_VIEWS} --n_refs ${BEST_N_REFS} \\
        --eval_strategy ${BEST_EVAL} \\
        --retrieval_strategy ${BEST_RETRIEVAL} \\
        --embedding_dir /app/embeddings \\
        --embedding_model ${EMBEDDING_MODEL} \\
        --seed \$SEED \\
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
for s in "$SCRIPT_DIR"/hybrid_*.sh; do
    echo "sbatch $(basename $s)"
    sbatch "$s"
    sleep 0.5
done
SUBMIT
chmod +x "${TARGET_DIR}/submit_all.sh"

echo "Generated $COUNTER scripts in ${TARGET_DIR}/"
echo "Submit: bash ${TARGET_DIR}/submit_all.sh"
