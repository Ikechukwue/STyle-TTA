#!/bin/bash
# ============================================================================
# Generate HPC SLURM Scripts: Style Transfer Method Evaluation
# ============================================================================
# Creates one SLURM job per style transfer method for parallel evaluation.
#
# Usage:
#   bash scripts/generate_hpc_style_transfer_eval.sh
#   # Then: bash scripts/generated/style_transfer_eval/submit_all.sh
# ============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

TARGET_DIR="${SCRIPT_DIR}/generated/style_transfer_eval"
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "========================================================================"
echo "Style Transfer Evaluation — HPC Script Generator"
echo "  Methods: ${#ALL_STYLE_METHODS[@]}"
echo "========================================================================"

for METHOD in "${ALL_STYLE_METHODS[@]}"; do
    # Diffusion methods need more GPU memory
    GPU_CONFIG="gpu:a40:1"; PARTITION="a40"; TIME="4:00:00"
    if [[ " ${DIFFUSION_METHODS[*]} " =~ " ${METHOD} " ]]; then
        GPU_CONFIG="gpu:a100:1"; PARTITION="a100"; TIME="8:00:00"
    fi

    cat > "${TARGET_DIR}/eval_${METHOD}.sh" << EOF
#!/bin/bash -l
#SBATCH --job-name=st-eval-${METHOD}
#SBATCH --gres=${GPU_CONFIG}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

CONTAINER=${HPC_CONTAINER}
DATA_PATH=${HPC_DATA_PATH}
WEIGHTS_DIR=${HPC_WEIGHTS_DIR}
OUTPUT_DIR=${HPC_OUTPUT_PATH}/style_transfer_eval

echo "========================================================================"
echo "Style Transfer Eval: ${METHOD}"
echo "Job: \$SLURM_JOB_ID | \$(date)"
echo "========================================================================"

mkdir -p \$OUTPUT_DIR
[ ! -f "\$CONTAINER" ] && echo "ERROR: Container not found" && exit 1

# Stage data
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
timeout 7h apptainer exec --nv \\
    --bind \$EFFECTIVE_DATA_PATH:/app/data \\
    --bind \$WEIGHTS_DIR:/app/weights \\
    --bind \$OUTPUT_DIR:/app/output \\
    \$CONTAINER \\
    python -m experiments.style_transfer_evaluation \\
        --data_path /app/data \\
        --weights_dir /app/weights \\
        --output_dir /app/output \\
        --content_dataset imagenet --content_split test_r \\
        --style_dataset imagenet --style_split train \\
        --n_content 20 --n_style 40 \\
        --input_size 256 --seed 42 \\
        --methods ${METHOD}

echo "Done: \$? | \$(date)"
EOF
    chmod +x "${TARGET_DIR}/eval_${METHOD}.sh"
done

# Submit-all script
cat > "${TARGET_DIR}/submit_all.sh" << 'SUBMIT'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for s in "$SCRIPT_DIR"/eval_*.sh; do
    echo "sbatch $(basename $s)"
    sbatch "$s"
    sleep 1
done
SUBMIT
chmod +x "${TARGET_DIR}/submit_all.sh"

echo "Generated ${#ALL_STYLE_METHODS[@]} scripts in ${TARGET_DIR}/"
echo "Submit: bash ${TARGET_DIR}/submit_all.sh"
