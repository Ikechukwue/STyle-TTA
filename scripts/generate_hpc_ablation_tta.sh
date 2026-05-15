#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: RetriStyle TTA Ablation Suite
# ============================================================================
# Three ablation axes on ImageNet → ImageNet-R:
#   A) Retrieval strategy (fix best eval + n_refs, sweep retrieval)
#   B) Aggregation strategy (fix best retrieval + n_refs, sweep eval)
#   C) n_refs sweep (fix best retrieval + eval, sweep n_refs)
#
# Default best settings (override via env):
#   BEST_RETRIEVAL=dino  BEST_EVAL=zero  BEST_N_REFS=16
#
# Sweeps across all classifiers × 3 seeds.
#
# Usage:
#   bash scripts/generate_hpc_ablation_tta.sh
#   # Override best:
#   BEST_RETRIEVAL=metric BEST_EVAL=tpt bash scripts/generate_hpc_ablation_tta.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

BEST_RETRIEVAL="${BEST_RETRIEVAL:-dino}"
BEST_EVAL="${BEST_EVAL:-zero}"
BEST_N_REFS="${BEST_N_REFS:-16}"

TARGET_DIR="${SCRIPT_DIR}/generated/ablation"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

TTA_METHOD="retristyle"
AUGMENTATION="none"
DATASET="imagenet"
SPLIT="test_abl"

echo "========================================================================"
echo "RetriStyle Ablation Suite — HPC Script Generator"
echo "  Best retrieval : ${BEST_RETRIEVAL}"
echo "  Best eval      : ${BEST_EVAL}"
echo "  Best n_refs    : ${BEST_N_REFS}"
echo "  Classifiers    : ${ALL_CLASSIFIERS[*]}"
echo "  Seeds          : ${ALL_SEEDS[*]}"
echo ""
echo "  A) Retrieval   : ${RETRIEVAL_STRATEGIES[*]}"
echo "  B) Eval        : ${EVAL_STRATEGIES[*]}"
echo "  C) n_refs      : ${N_REFS_VALUES[*]}"
echo "========================================================================"

COUNTER=0

# Helper: generate one SLURM script
# Args: SUBDIR CLF EVAL RETR NREFS SEED TAG
gen_script() {
    local SUBDIR=$1 CLF=$2 EVAL=$3 RETR=$4 NREFS=$5 SEED=$6 TAG=$7
    is_pretrained "$CLF" && return 0
    local CLF_SAFE=$(safe_name "$CLF")

    local TIER=$(gpu_tier_for_nrefs "$NREFS")
    local GPU_CONFIG=$(echo "$TIER" | awk '{print $1}')
    local PARTITION=$(echo "$TIER" | awk '{print $2}')

    local WEIGHTS_LINE='WEIGHTS_PATH="pretrained"'
    if is_pretrained "$CLF"; then
        WEIGHTS_LINE='WEIGHTS_PATH=$WORK/retristyle/data/imagenet-$CLASSIFIER-random_flip-random_resized_crop-seed42.pth'
    fi

    local SCRIPT="${TARGET_DIR}/${SUBDIR}/abl_${CLF_SAFE}_${RETR}_${EVAL}_nr${NREFS}_s${SEED}.sh"
    mkdir -p "$(dirname "$SCRIPT")"

    cat > "$SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=abl-${TAG}-${CLF_SAFE}-s${SEED}
#SBATCH --gres=${GPU_CONFIG}
#SBATCH --partition=${PARTITION}
#SBATCH --time=3:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

TTA_METHOD="${TTA_METHOD}"
EVAL_STRATEGY="${EVAL}"
RETRIEVAL_STRATEGY="${RETR}"
CLASSIFIER="${CLF}"
AUGMENTATION="${AUGMENTATION}"
DATASET="${DATASET}"
SEED=${SEED}
SPLIT="${SPLIT}"
N_REFS=${NREFS}
${WEIGHTS_LINE}

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
OUTPUT_PATH=${HPC_OUTPUT_PATH}
EMBEDDING_DIR=${HPC_EMBEDDING_DIR}
HF_MODELS_CACHE=${HPC_HF_CACHE}
TORCH_MODELS_CACHE=${HPC_TORCH_CACHE}
MODEL_DIR=${HPC_MODEL_DIR}
LIVE_CODE=\$HOME/retristyle/retristyle

echo "Ablation ${TAG}: \$CLASSIFIER | retr=\$RETRIEVAL_STRATEGY | eval=\$EVAL_STRATEGY | nr=\$N_REFS | seed=\$SEED"
echo "Job: \$SLURM_JOB_ID | \$(date)"
mkdir -p \$OUTPUT_PATH \$EMBEDDING_DIR

[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

GPU_DEVICES="\${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=\$(echo \$GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=\$(printf "/app/configs/gpu_%02d.yaml" \$GPU_COUNT)

if [[ -n "\$TMPDIR" ]]; then
    echo "Unpacking Source ImageNet to SSD..."
    rsync -ahW --progress \$HPCVAULT/data/${DATASET}_${SPLIT}.tar \$TMPDIR/
    tar -xf \$TMPDIR/${DATASET}_${SPLIT}.tar -C \$TMPDIR/
    EFFECTIVE_DATA_PATH=\$TMPDIR/data

    echo "Unpacking Augmented Cache to SSD..."
    TARGET_EXTRACT_DIR="\$TMPDIR/data/augmented_cache/\$SEED/\$CLASSIFIER/\$TTA_METHOD"
    LOCAL_CACHE=\$TMPDIR/data/augmented_cache
    mkdir -p \$LOCAL_CACHE \$TARGET_EXTRACT_DIR

    TAR_FILE="${BEST_RETRIEVAL}_${BEST_N_REFS}_${DATASET}_${SPLIT}_s${SEED}.tar"
    rsync -ahW --progress \$HPCVAULT/augmented_cache/${SPLIT}/complete/\$TAR_FILE \$TMPDIR/
    tar -xf \$TMPDIR/\$TAR_FILE -C \$TARGET_EXTRACT_DIR
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
timeout 2h apptainer exec --nv \\
    --pwd /app \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$OUTPUT_PATH:/app/results \\
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
        --retrieval_strategy \$RETRIEVAL_STRATEGY \\
        --n_refs \$N_REFS --n_views ${DEFAULT_N_VIEWS} \\
        --style_batch_size ${STYLE_BATCH_SIZE} \\
        --embedding_model ${EMBEDDING_MODEL} \\
        --augmented_cache /app/data/augmented_cache \\
        --embedding_dir /app/data/embeddings \\
        --seed \$SEED \\
        --output_path /app/results

EXIT_CODE=\$?
echo "Done: \$EXIT_CODE | \$(date)"
[[ \$EXIT_CODE -eq 124 ]] && sbatch "\${BASH_SOURCE[0]}"

exit \$EXIT_CODE
EOF
    chmod +x "$SCRIPT"
    COUNTER=$((COUNTER + 1))
}

# ============================================================================
# A) Retrieval Strategy Ablation
# ============================================================================
echo "A) Generating retrieval strategy ablation..."
for CLF in "${ALL_CLASSIFIERS[@]}"; do
for RETR in "${RETRIEVAL_STRATEGIES[@]}"; do
for SEED in "${ALL_SEEDS[@]}"; do
    gen_script "retrieval" "$CLF" "$BEST_EVAL" "$RETR" "$BEST_N_REFS" "$SEED" "retr"
done; done; done

# ============================================================================
# B) Aggregation Strategy Ablation
# ============================================================================
echo "B) Generating eval strategy ablation..."
for CLF in "${ALL_CLASSIFIERS[@]}"; do
for EVAL in "${EVAL_STRATEGIES[@]}"; do
for SEED in "${ALL_SEEDS[@]}"; do
    gen_script "eval_strategy" "$CLF" "$EVAL" "$BEST_RETRIEVAL" "$BEST_N_REFS" "$SEED" "eval"
done; done; done

# ============================================================================
# C) n_refs Sweep
# ============================================================================
echo "C) Generating n_refs sweep..."
for CLF in "${ALL_CLASSIFIERS[@]}"; do
for NR in "${N_REFS_VALUES[@]}"; do
for SEED in "${ALL_SEEDS[@]}"; do
    gen_script "n_refs" "$CLF" "$BEST_EVAL" "$BEST_RETRIEVAL" "$NR" "$SEED" "nrefs"
done; done; done

# ============================================================================
# Submit-all
# ============================================================================
cat > "${TARGET_DIR}/submit_all.sh" << 'SUBMIT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for dir in retrieval eval_strategy n_refs; do
    [[ -d "$SCRIPT_DIR/$dir" ]] || continue
    echo "=== Submitting ablation: $dir ==="
    for s in "$SCRIPT_DIR/$dir"/abl_*.sh; do
        echo "  sbatch $(basename $s)"
        sbatch "$s"
        sleep 0.5
    done
done
SUBMIT
chmod +x "${TARGET_DIR}/submit_all.sh"

echo ""
echo "Generated $COUNTER ablation scripts in ${TARGET_DIR}/"
echo "Submit: bash ${TARGET_DIR}/submit_all.sh"
