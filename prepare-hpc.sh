#!/bin/bash
# Script to prepare and deploy style_tta to HPC cluster

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="style_tta"
DEFAULT_TARGET="production"  # Only production target available

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Prepare style_tta Docker image for HPC deployment

OPTIONS:
    -t, --target TARGET      Build target (only 'production' supported)
                            Default: production
    -o, --output FILE       Output file name for Docker archive
                            Default: ${PROJECT_NAME}-TARGET.tar
    -s, --apptainer         Also convert to Apptainer image
    --no-build             Skip building, use existing image
    --pytorch VERSION      PyTorch version (default: 2.9.1)
    --cuda VERSION         CUDA version (default: 12.8.0)
    --python VERSION       Python version (default: 3.13)
    -h, --help             Show this help message

EXAMPLES:
    # Build production image and save as tar
    $0 --target production

    # Build and convert to Apptainer
    $0 --target production --apptainer

    # Use existing image, just export
    $0 --target production --no-build

    # Custom versions
    $0 --pytorch 2.9.1 --cuda 12.8.0 --python 3.13

EOF
}

# Parse command line arguments
TARGET="$DEFAULT_TARGET"
OUTPUT=""
BUILD=true
CONVERT_APPTAINER=false
PYTORCH_VERSION="2.9.1"
CUDA_VERSION="12.8.0"
PYTHON_VERSION="3.13"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -s|--apptainer)
            CONVERT_APPTAINER=true
            shift
            ;;
        --no-build)
            BUILD=false
            shift
            ;;
        --pytorch)
            PYTORCH_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cudnn)
            # Deprecated: cuDNN version now included in CUDA image tag
            echo "Warning: --cudnn flag is deprecated (cuDNN included in CUDA image)"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate target
if [[ "$TARGET" != "production" ]]; then
    print_warn "Only 'production' target is supported. Using production."
    TARGET="production"
fi

# Set output file name
if [ -z "$OUTPUT" ]; then
    OUTPUT="${PROJECT_NAME}-${TARGET}.tar"
fi

IMAGE_TAG="${PROJECT_NAME}:${TARGET}"
SIF_OUTPUT="${OUTPUT%.tar}.sif"

print_info "Starting HPC deployment preparation"
print_info "Target: $TARGET"
print_info "Docker image: $IMAGE_TAG"
print_info "Output file: $OUTPUT"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Build Docker image
if [ "$BUILD" = true ]; then
    print_info "Building Docker image..."
    
    BUILD_ARGS=(
        "--target" "$TARGET"
        "--build-arg" "PYTORCH_VERSION=$PYTORCH_VERSION"
        "--build-arg" "CUDA_VERSION=$CUDA_VERSION"
        "--build-arg" "PYTHON_VERSION=$PYTHON_VERSION"
        "-t" "$IMAGE_TAG"
    )
    
    if ! docker build "${BUILD_ARGS[@]}" "$SCRIPT_DIR"; then
        print_error "Docker build failed"
        exit 1
    fi
    
    print_info "Docker image built successfully"
else
    print_info "Skipping build, using existing image"
    
    # Check if image exists
    if ! docker image inspect "$IMAGE_TAG" &> /dev/null; then
        print_error "Image $IMAGE_TAG does not exist"
        print_error "Run without --no-build to build it"
        exit 1
    fi
fi

# Save Docker image to tar
print_info "Saving Docker image to $OUTPUT..."
if ! docker save "$IMAGE_TAG" -o "$OUTPUT"; then
    print_error "Failed to save Docker image"
    exit 1
fi

print_info "Docker image saved to $OUTPUT"
print_info "Size: $(du -h "$OUTPUT" | cut -f1)"

# Convert to Apptainer if requested
if [ "$CONVERT_APPTAINER" = true ]; then
    print_info "Converting to Apptainer image..."
    
    # Check if Apptainer/Singularity is available
    if command -v apptainer &> /dev/null; then
        print_info "Using apptainer command"
        CONTAINER_CMD="apptainer"
    elif command -v singularity &> /dev/null; then
        print_info "Using singularity command (Apptainer-compatible)"
        CONTAINER_CMD="singularity"
    else
        print_warn "Neither apptainer nor singularity found in PATH"
        print_warn "You'll need to convert to Apptainer on NHR@FAU cluster"
        print_warn "Run this command on HPC frontend node:"
        print_warn "  apptainer build $SIF_OUTPUT docker-archive://$OUTPUT"
    fi
    
    if [ -n "$CONTAINER_CMD" ]; then
        if $CONTAINER_CMD build "$SIF_OUTPUT" "docker-archive://$OUTPUT"; then
            print_info "Apptainer image created: $SIF_OUTPUT"
            print_info "Size: $(du -h "$SIF_OUTPUT" | cut -f1)"
        else
            print_error "Failed to create Apptainer image"
            print_warn "You may need to convert it on NHR@FAU cluster"
        fi
    fi
fi

# Create deployment instructions
INSTRUCTIONS_FILE="${PROJECT_NAME}-${TARGET}-deployment.txt"
cat > "$INSTRUCTIONS_FILE" << EOF
===============================================================================
style_tta HPC Deployment Instructions
===============================================================================

IMAGE DETAILS:
  Target: $TARGET
  Docker image: $IMAGE_TAG
  Archive file: $OUTPUT
  $([ "$CONVERT_APPTAINER" = true ] && echo "Apptainer file: $SIF_OUTPUT")
  
  Build configuration:
    - PyTorch: $PYTORCH_VERSION
    - CUDA: $CUDA_VERSION (with cuDNN 9.x included)
    - Python: $PYTHON_VERSION
    - System packages: ImageMagick/libmagickwand (for medmnistc augmentations)

===============================================================================
DEPLOYMENT STEPS:
===============================================================================

1. TRANSFER TO NHR@FAU CLUSTER:

   # Transfer to NHR@FAU cluster (use csnhr.nhr.fau.de dialog server)
   rsync -avz --progress $OUTPUT USERNAME@csnhr.nhr.fau.de:\$WORK/style_tta/

   $([ -f "$SIF_OUTPUT" ] && echo "rsync -avz --progress $SIF_OUTPUT USERNAME@csnhr.nhr.fau.de:\$WORK/style_tta/")

2. CONVERT TO APPTAINER (if not already done):

   On NHR@FAU frontend node (e.g., Alex, Fritz, Woody, Meggie):
   
   # Apptainer is available by default, no module load needed
   cd \$WORK/style_tta
   apptainer build $SIF_OUTPUT docker-archive://$OUTPUT

3. PREPARE DATA DIRECTORIES ON NHR@FAU:

   # Use \$WORK for active data (no backup, large quota)
   mkdir -p \$WORK/style_tta/data
   mkdir -p \$WORK/style_tta/checkpoints
   mkdir -p \$WORK/style_tta/outputs
   mkdir -p \$WORK/style_tta/logs
   
   # Store important code and final results in \$HOME (backed up, 50GB limit)
   mkdir -p \$HOME/style_tta/scripts
   mkdir -p \$HOME/style_tta/results
   
   # For long-term archive use \$HPCVAULT (backed up, 500GB)
   mkdir -p \$HPCVAULT/style_tta/archive
   
   # Note: Local development uses different paths:
   # - Code: /home/staff/sdoerric/research/style_tta/
   # - Data: /data/local/style_tta/{data,checkpoints,results,logs}

4. SET ENVIRONMENT VARIABLES:

   export WANDB_API_KEY="your_wandb_api_key"
   export WANDB_ENTITY="ofu-xai"
   export WANDB_PROJECT="style_tta"

5. TEST THE IMAGE:

   # Test basic functionality
   apptainer exec \$WORK/style_tta/$SIF_OUTPUT python -c "import torch; print(torch.cuda.is_available())"
   
   # Test GPU access (requires --nv flag on GPU node)
   apptainer exec --nv \$WORK/style_tta/$SIF_OUTPUT nvidia-smi

6. RUN TRAINING:

   # Interactive session on Alex GPU cluster
   salloc --partition=a100 --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=2:00:00
   
   # Once allocated, run training (--nv flag required for GPU support):
   apptainer exec --nv \\
     --bind \$WORK/style_tta/data:/app/data \\
     --bind \$WORK/style_tta/checkpoints:/app/checkpoints \\
     --bind \$WORK/style_tta/outputs:/app/outputs \\
     --bind \$WORK/style_tta/logs/wandb:/app/wandb \\
     \$WORK/style_tta/$SIF_OUTPUT \\
     accelerate launch --config_file /app/configs/auto_gpu.yaml \\
       /app/experiments/reference_methods/pretrain.py \\
       --method adain \\
       --dataset_source epistr \\
       --dataset_reference epistr

   # Or submit SLURM job (see scripts/pretraining/adain/adain_epistr_hpc.sh)

===============================================================================
USEFUL COMMANDS FOR NHR@FAU:
===============================================================================

# Check Apptainer version
apptainer --version

# Inspect image
apptainer inspect \$WORK/style_tta/$SIF_OUTPUT

# Shell into container
apptainer shell \\
  --bind \$WORK/style_tta/data:/app/data \\
  \$WORK/style_tta/$SIF_OUTPUT

# Check GPU access (on GPU node, requires --nv flag)
apptainer exec --nv \$WORK/style_tta/$SIF_OUTPUT nvidia-smi

# List available Python packages
apptainer exec \$WORK/style_tta/$SIF_OUTPUT pip list

# Check filesystem quotas
shownicerquota.pl

# View available GPU partitions
sinfo -p a100       # A100 40GB GPUs
sinfo -p a100_80gb  # A100 80GB GPUs (if available)

# Set Apptainer cache location (add to ~/.bashrc)
export APPTAINER_CACHEDIR=\$WORK/.apptainer/cache

===============================================================================
TROUBLESHOOTING:
===============================================================================

- If GPU not detected: Ensure --nv flag is used and you're on a GPU node
- If out of memory: Increase --mem in SLURM or reduce batch size
- If permission denied: Check file permissions and bind mounts
- If missing dependencies: Rebuild image with updated requirements
- If accelerate fails to detect GPUs: Check CUDA_VISIBLE_DEVICES is set

Generated: $(date)
EOF

print_info "Deployment instructions saved to $INSTRUCTIONS_FILE"

# Summary
echo ""
echo "==============================================================================="
print_info "HPC deployment preparation complete!"
echo "==============================================================================="
echo ""
echo "Files created:"
echo "  - Docker archive: $OUTPUT"
[ -f "$SIF_OUTPUT" ] && echo "  - Apptainer image: $SIF_OUTPUT"
echo "  - Instructions: $INSTRUCTIONS_FILE"
echo ""
echo "Next steps:"
echo "  1. Review the deployment instructions: cat $INSTRUCTIONS_FILE"
echo "  2. Transfer files to your HPC cluster using rsync"
echo "  3. Follow the deployment steps in the instructions"
echo "  4. Submit training job: sbatch scripts/pretraining/adain/adain_epistr_hpc.sh"
echo ""
echo "==============================================================================="
