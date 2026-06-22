#!/bin/bash
# ============================================================================
# Common configuration for all retristyle scripts.
# Source this file in other scripts: source "$(dirname "$0")/common.sh"
# ============================================================================

# ── Project root (relative to scripts/) ──
PROJECT_ROOT="/home/stud/nemmler/retristyle"

# ── Default paths (local machine) ──
DATA_PATH="${DATA_PATH:-/data/local/retristyle/data}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/results}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/data/local/retristyle/models/style_transfer}"
EMBEDDING_DIR="${EMBEDDING_DIR:-${PROJECT_ROOT}/data/embeddings}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/data/models}"
AUG_DIR="${AUG_DIR:-${PROJECT_ROOT}/data/augmented_cache}"
# ── HPC cluster paths (NHR@FAU) ──
HPC_CONTAINER='$WORK/retristyle/retristyle-production.sif'
HPC_DATA_PATH='$WORK/retristyle/data'
HPC_OUTPUT_PATH='$WORK/retristyle/results'
HPC_WEIGHTS_DIR='$WORK/retristyle/models'
HPC_EMBEDDING_DIR='$HPCVAULT/retristyle/embeddings'
HPC_MODEL_DIR='$WORK/retristyle/models'
HPC_HF_CACHE='$WORK/model_cache/hf'
HPC_TORCH_CACHE='$WORK/model_cache/torch'
HPC_LIVE_CODE='$HPCVAULT/snapshots/retristyle_20260518'
# ── Default experiment settings ──
DEFAULT_SEED=71397589
ALL_SEEDS=(71397589 133560673 265017005)

# ── Classifiers ──
CNN_CLASSIFIERS=("resnet18" "densenet121")
VIT_CLASSIFIERS=("vit_base_patch16_224" "swin_base_patch4_window7_224")
VLM_CLASSIFIERS=("ViT-B-16" "ViT-B-16@Zero")
FM_CLASSIFIERS=("dinov2_vitb14" "vit_base_patch16_dinov3_lvd1689m")
ALL_CLASSIFIERS=("${CNN_CLASSIFIERS[@]}" "${VIT_CLASSIFIERS[@]}" "${VLM_CLASSIFIERS[@]}" "${FM_CLASSIFIERS[@]}")
PRETRAINED_CLASSIFIERS=("ViT-B-16" "dinov2_vitb14" "vit_base_patch16_dinov3_lvd1689m")

# ── ImageNet datasets (primary) ──
IMAGENET_DATASET="imagenet"
IMAGENET_TRAIN_SPLIT="train"
IMAGENET_TEST_SPLITS=("test_r" "test_a" "test_sketch" "test_v2" "testing" "test_r_c26")

# ── All available datasets (uncomment when extending) ──
# NATURAL_DATASETS=("pacs" "vlcs" "domainnet" "office_home" "terra_incognita")
# MEDICAL_DATASETS=("camelyon17wilds" "epistr" "peripheral_blood" "fitzpatrick17k" "retina")

# ── Retrieval / TTA settings ──
EMBEDDING_MODEL="vit_base_patch16_dinov3.lvd1689m"
RETRIEVAL_STRATEGIES=("random" "balanced_random" "dino") #"metric" "balanced_metric")
EVAL_STRATEGIES=("vanilla" "zero" "tpt")
N_REFS_VALUES=(2 4 8 16 32)
DEFAULT_N_VIEWS=16
STYLE_BATCH_SIZE=8

# ── Hybrid TTA ratios ──
GEO_FRACS=("1.0" "0.75" "0.5" "0.25" "0.0")

# ── Style transfer methods ──
TRAINED_METHODS=("adain" "adaattn" "aespanet" "artflow" "cast" "efdm" "iecontrast" "mast" "sanet" "styleformer" "stytr2")
DIFFUSION_METHODS=("styleid" "diffstyle")
PHOTO_METHODS=("modflows" "wct2" "deeppreset" "photonas")
ALL_STYLE_METHODS=("${TRAINED_METHODS[@]}" "${DIFFUSION_METHODS[@]}" "${PHOTO_METHODS[@]}")

# ── Training augmentations ──
TRAINING_AUGMENTATIONS=("none" "color_jitter" "rand_augment" "trivial_augment" "aug_mix")

# ── DomainBed methods ──
DOMAINBED_METHODS=("ERM" "IB_ERM" "Mixup" "RSC" "SD" "SelfReg")

# ── SDG methods ──
SDG_METHODS=("MixStyle")

# ============================================================================
# Helper functions
# ============================================================================

is_pretrained() {
    local clf=$1
    for pc in "${PRETRAINED_CLASSIFIERS[@]}"; do
        [[ "$clf" == "$pc" ]] && return 0
    done
    return 1
}

safe_name() {
    echo "$1" | tr '/' '_' | tr '-' '_'
}

gpu_tier_for_nrefs() {
    local n=$1
    case $n in
        2|4)  echo "gpu:a100:1 a100" ;;
        8|16) echo "gpu:a100:2 a100" ;;
        32|64) echo "gpu:a100:4 a100" ;;
        *)    echo "gpu:a100:1 a100" ;;
    esac
}

# Generate the standard HPC preamble (proxy, validation, GPU config)
hpc_preamble() {
    cat << 'EOF'

# Proxy for NHR@FAU compute nodes
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy

# Validation
echo "========================================================================"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | $(date)"
echo "========================================================================"

[ -z "$WORK" ] && echo "ERROR: \$WORK not set" && exit 1
[ ! -f "$CONTAINER" ] && echo "ERROR: Container not found: $CONTAINER" && exit 1

mkdir -p $OUTPUT_PATH
mkdir -p $WORK/.apptainer/cache

# GPU configuration
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_COUNT=$(echo $GPU_DEVICES | tr ',' '\n' | wc -l)
ACCELERATE_CONFIG=$(printf "/app/configs/gpu/gpu_%02d.yaml" $GPU_COUNT)
echo "GPUs: $GPU_COUNT | Config: $ACCELERATE_CONFIG"
EOF
}

# Generate TMPDIR staging for ImageNet data
hpc_stage_imagenet() {
    cat << 'EOF'

# Stage data to $TMPDIR for fast I/O
if [[ -n "$TMPDIR" ]]; then
    echo "Unpacking data to $TMPDIR..."
    mkdir -p $TMPDIR/data
    # Unpack your tarball directly into the node's local SSD
    tar -xf $WORK/retristyle/retristyle_data.tar -C $TMPDIR/
    EFFECTIVE_DATA_PATH=$TMPDIR/data
    echo "Staging complete."
else
    EFFECTIVE_DATA_PATH=$DATA_PATH
fi
EOF
}
