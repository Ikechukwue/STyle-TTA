# Getting Started with retristyle: Complete Setup Guide

This guide provides step-by-step instructions for setting up retristyle from scratch, including local development, Docker containerization, and HPC cluster deployment.

## Local Machine Setup

### Step 1: Install Docker

Docker is required to create reproducible containers that work across different operating systems.

```bash
# Remove old versions and conflicting packages
sudo apt-get remove docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc

# Set up Docker's apt repository
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update

# Install Docker Engine
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Verify installation
sudo docker run hello-world

# Add your user to docker group (to run without sudo)
sudo usermod -aG docker $USER

# Apply group changes (or log out and back in)
newgrp docker

# Test without sudo
docker --version
docker run hello-world
```

**Expected output:** You should see "Hello from Docker!" message.

### Step 2: Install NVIDIA Docker Runtime (Optional, for GPU support)

If you have an NVIDIA GPU and want to test GPU training locally:

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install -y curl gnupg2

# Configure the NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
sudo apt-get update

# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi  # cuda 12.8
# docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi  # cuda 12.9
```

**Expected output:** You should see your GPU information from `nvidia-smi`.

### Step 3: Install Apptainer (for HPC compatibility)

Apptainer (formerly Singularity) is used to convert Docker images for HPC clusters.

```bash
# On Ubuntu based systems, first ensure software-properties-common is installed
sudo apt-get update
sudo apt-get install -y software-properties-common

# Add Apptainer PPA repository
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update

# Install Apptainer
sudo apt-get install -y apptainer

# Verify installation
apptainer --version
```

**Expected output:** `apptainer version 1.x.x` or similar.

### Step 4: Clone the retristyle Repository

```bash
# Navigate to your research directory
cd /home/staff/sdoerric/research/

# If not already cloned, clone the repository
# git clone <repository-url> retristyle

# Navigate to project directory
cd retristyle

# Verify you're in the right place
ls -la
```

**Expected output:** You should see `Dockerfile`, `requirements.txt`, `setup.py`, etc.

### Step 5: Set Up Data Directories

Create the directory structure for datasets, checkpoints, and outputs:

```bash
# Create data directories on your local HDD
sudo mkdir -p /data/local/colorist/{data,checkpoints,results,logs}

# Set ownership to your user
sudo chown -R $USER:$USER /data/local/retristyle

# Create subdirectories
mkdir -p /data/local/colorist/data/
mkdir -p /data/local/colorist/checkpoints/
mkdir -p /data/local/retristyle/results/
mkdir -p /data/local/retristyle/logs/

# Verify structure
tree -L 3 /data/local/colorist
```

### Step 6: Configure Environment Variables

Set up your Weights & Biases (WandB) credentials:

```bash
# Add to your ~/.bashrc for persistence
# echo 'export WANDB_API_KEY="[YOUR_KEY]"' >> ~/.bashrc
# echo 'export WANDB_ENTITY="[YOUR_ENTITY]"' >> ~/.bashrc
# echo 'export WANDB_PROJECT="retristyle"' >> ~/.bashrc

# Apply changes
source ~/.bashrc

# Verify
echo $WANDB_API_KEY
```

**Note:** Get your WandB API key from https://wandb.ai/authorize

---

## Building Docker Images

### Step 7: Build Production Docker Image

The production image contains all dependencies and is optimized for cluster deployment.

```bash
# Navigate to project root
cd /home/stud/nemmler/retristyle

# Build production image (this takes 10-20 minutes)
# Using versions matching your workstation: CUDA 12.8, Python 3.13, PyTorch 2.9.1
docker build \
    --target production \
    --build-arg PYTORCH_VERSION=2.9.1 \
    --build-arg CUDA_VERSION=12.8.0 \
    --build-arg PYTHON_VERSION=3.13 \
    -t retristyle:production \
    .

# Verify image was created
docker images | grep retristyle
```

**Expected output:**
```
retristyle     production     <image-id>     X minutes ago     ~7GB
```

### Step 8: Create Apptainer Image

Convert the Docker image to Apptainer format for HPC deployment.

```bash
# Method 1: Using the automated script (recommended)
./prepare-hpc.sh --target production --apptainer

# This creates:
# - retristyle-production.tar (Docker archive)
# - retristyle-production.sif (Apptainer image)
# - retristyle-production-deployment.txt (instructions)

# Method 2: Manual conversion
# Save Docker image to tar
docker save retristyle:production -o retristyle-production.tar

# Convert to Apptainer SIF
apptainer build retristyle-production.sif docker-archive://retristyle-production.tar

# Verify Apptainer image
ls -lh retristyle-production.sif
apptainer inspect retristyle-production.sif
```

**Expected output:** `retristyle-production.sif` file (~7-8GB)

---

## Cluster Deployment

### Step 12: Transfer Files to HPC Cluster

Transfer the Apptainer image and necessary files to NHR@FAU.

```bash
# Set your cluster username and connection details
# Always use csnhr.nhr.fau.de as the remote host (NHR@FAU dialog server)
# Tiny GPU
export CLUSTER_USER="[USER]"
export CLUSTER_HOST="csnhr.nhr.fau.de"
export CLUSTER_WORK="/home/woody/barz/${CLUSTER_USER}"
# If you need to get the cluster work: ssh ${CLUSTER_USER}@${CLUSTER_HOST} "echo $WORK"

# Transfer Apptainer image to $WORK directory (this takes 20-30 minutes)
# Using rsync with --progress for progress bar and resumability
# Note: rsync automatically creates the destination directory
rsync -avz --progress retristyle-production.sif ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_WORK}/retristyle/

# Alternative: Transfer Docker tar and convert on cluster
# rsync -avz --progress retristyle-production.tar ${CLUSTER_USER}@${CLUSTER_HOST}:$WORK/retristyle/

# Transfer SLURM scripts to $HOME/retristyle/scripts (relative path from $HOME)
# Relative paths without leading '/' are automatically relative to $HOME
rsync -avz --progress scripts/ ${CLUSTER_USER}@${CLUSTER_HOST}:retristyle/scripts/
```

### Step 13: Set Up Cluster Directories

SSH to the cluster and create the required directory structure:

```bash
# SSH to cluster
ssh ${CLUSTER_USER}@${CLUSTER_HOST}

# On cluster: Create directory structure
# $WORK: Active data, no backup, large quota (500GB-1TB)
mkdir -p $WORK/colorist/{data,checkpoints,outputs,logs}
mkdir -p $WORK/colorist/checkpoints/reference_methods/adain
mkdir -p $WORK/colorist/logs/slurm

# $HOME: Scripts and final results, backed up, 50GB quota
mkdir -p $HOME/retristyle/{scripts,results}
```

### Step 14: Transfer Datasets to Cluster

Transfer datasets and pre-trained models from your local machine:

```bash
# From your local machine, transfer datasets
# This can take hours depending on dataset size - consider using rsync for resumability
# Note: rsync automatically creates destination directories

# Make sure that the environment variables are still exported
# Tiny GPU
export CLUSTER_USER="[USER]"
export CLUSTER_HOST="csnhr.nhr.fau.de"
export CLUSTER_WORK="/home/woody/barz/${CLUSTER_USER}"

# Transfer data
rsync -avz --progress /data/local/colorist/data ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_WORK}/colorist/

# Transfer pre-trained model weights
rsync -avz --progress /data/local/colorist/checkpoints ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_WORK}/colorist/
```

### Step 15: Convert Docker to Apptainer on Cluster (if needed)

If you transferred the `.tar` file instead of `.sif`:

```bash
# On cluster
ssh ${CLUSTER_USER}@${CLUSTER_HOST}

# Set Apptainer cache directory (important for large files)
export APPTAINER_CACHEDIR=$WORK/.apptainer/cache
export APPTAINER_TMPDIR=$WORK/.apptainer/tmp
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR

# Add to ~/.bashrc for persistence
echo 'export APPTAINER_CACHEDIR=$WORK/.apptainer/cache' >> ~/.bashrc
echo 'export APPTAINER_TMPDIR=$WORK/.apptainer/tmp' >> ~/.bashrc

# Convert Docker archive to Apptainer image (takes 10-15 minutes)
cd $WORK/colorist
apptainer build colorist-production.sif docker-archive://colorist-production.tar

# Verify conversion
ls -lh colorist-production.sif
apptainer inspect colorist-production.sif
```

### Step 16: Test Container on Cluster

Before submitting batch jobs, test the container interactively:

```bash
# On cluster: Test basic functionality
apptainer exec $WORK/retristyle/retristyle-production.sif python -c "import torch; import retristyle; print('Imports successful')"

# Request interactive GPU session for testing
salloc --gres=gpu:a100:1 --partition=a100 --cpus-per-task=8 --time=1:00:00

# Once allocated to a GPU node, test GPU access
apptainer exec $WORK/retristyle/retristyle-production.sif python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Test data loading
apptainer exec --bind $WORK/colorist/data:/app/data:ro $WORK/retristyle/retristyle-production.sif python -c "import os; print('Data files:', os.listdir('/app/data'))"

# Exit interactive session
exit
```

**Expected outputs:**
- `Imports successful`
- `CUDA: True, Device: NVIDIA A100-SXM4-80GB` (or similar)
- List of your data directories

---

## Running Training on Cluster

### Step 17: Configure SLURM Scripts

Copy and customize the SLURM scripts for your training needs:

```bash
# On cluster
cd $HOME/retristyle/scripts

# Copy the pretraining script
cp ../../../research/retristyle/scripts/pretraining/adain/adain_epistr_hpc.sh ./

# Edit the script to verify paths
nano adain_epistr_hpc.sh
```

**Key settings to verify in the SLURM script:**

```bash
#!/bin/bash
#SBATCH --job-name=adain-epistr           # Job name
#SBATCH --partition=a100-80gb             # GPU partition
#SBATCH --gres=gpu:4                      # Number of GPUs
#SBATCH --cpus-per-task=32                # CPU cores
#SBATCH --mem=256G                        # Memory
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=$WORK/retristyle/logs/slurm/pretraining-adain-epistr_%j.out

# Container and paths
CONTAINER="$WORK/retristyle/retristyle-production.sif"
DATA_DIR="$WORK/colorist/data"
CHECKPOINT_DIR="$WORK/colorist/checkpoints"
OUTPUT_DIR="$WORK/colorist/outputs"
WANDB_DIR="$WORK/colorist/logs/wandb"

# WandB configuration
# export WANDB_API_KEY="your_key_here"
# export WANDB_ENTITY="ofu-xai"
# export WANDB_PROJECT="retristyle"
```

### Step 18: Submit Training Job

Submit your first training job:

```bash
# On cluster
cd $HOME/retristyle/scripts

# Submit job
sbatch adain_epistr_hpc.sh

# Check job status
squeue -u $USER

# View job details
scontrol show job <JOBID>
```

**Expected output:** Job ID assigned and job enters queue.

### Step 19: Monitor Training

Monitor your training job in several ways:

```bash
# Check job queue status
squeue -u $USER

# View live training output
tail -f $WORK/colorist/logs/slurm/pretraining-adain-epistr_*.out

# Monitor GPU usage (if job is running)
ssh <node-where-job-runs>  # Get node name from squeue
nvidia-smi

# Alternatively when node is not know
srun --pty --overlap --jobid <YOUR-JOBID> bash # attach to your running job, get job ID from squeue
nvidia-smi

# Check WandB logs (if online mode)
# Visit: https://wandb.ai/ofu-xai/retristyle

# Check checkpoint creation
ls -lh $WORK/colorist/checkpoints/reference_methods/adain/

# Check disk usage
du -sh $WORK/colorist/*
shownicerquota.pl
```

### Step 20: Handle Long-Running Jobs

For jobs longer than 24 hours, the SLURM script includes auto-restart:

```bash
# The script automatically handles timeouts:
# - Uses `timeout 23h` to stop before 24h limit
# - Checks exit code 124 (timeout)
# - Resubmits job automatically via `sbatch`

# Monitor across restarts
watch -n 60 'squeue -u $USER'

# Training will resume from last checkpoint automatically
# WandB run ID is preserved in checkpoint metadata
```

### Step 21: Fetch models and checkpoints from the cluster

To copy the trained models and checkpoints back from the cluster towards the local workstation

```bash
# Set your cluster username and connection details
# Always use csnhr.nhr.fau.de as the remote host (NHR@FAU dialog server)
# Tiny GPU
export CLUSTER_USER="[USER]"
export CLUSTER_HOST="csnhr.nhr.fau.de"
export CLUSTER_WORK="/home/woody/barz/${CLUSTER_USER}"

# Transfer model weights from cluster to local workstation
# Using rsync with --progress for progress bar and resumability
# Note: rsync automatically creates the destination directory
rsync -avz --progress ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_WORK}/colorist/models /data/local/colorist/


# Fetch TTA results
rsync -avz --progress ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_WORK}/retristyle/results/tta_inference /data/local/retristyle/results/
```

---

## Troubleshooting

### Issue: Docker build fails with "no space left on device"

**Solution:**
```bash
# Clean up Docker
docker system prune -a --volumes

# Check disk space
df -h

# Free up space if needed
```

### Issue: Permission denied when accessing /data/local/colorist

**Solution:**
```bash
# Fix ownership
sudo chown -R $USER:$USER /data/local/colorist

# Verify permissions
ls -la /data/local/colorist
```

### Issue: GPU not detected in Docker container

**Solution:**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# If fails, reinstall NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: Apptainer conversion fails with "no space left"

**Solution:**
```bash
# Set temporary directory to location with more space
export APPTAINER_TMPDIR=$WORK/.apptainer/tmp
export APPTAINER_CACHEDIR=$WORK/.apptainer/cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# Try conversion again
apptainer build retristyle-production.sif docker-archive://retristyle-production.tar
```

### Issue: Job stuck in queue on cluster

**Solution:**
```bash
# Check partition status
sinfo -p a100-80gb

# See why job is pending
squeue -u $USER --start

# Consider using TinyGPU for testing (shorter wait)
#SBATCH --partition=a100  # In SLURM script
```

### Issue: "File not found" errors in container

**Solution:**
```bash
# Verify bind mounts are correct
apptainer exec \
    --bind $WORK/colorist/data:/app/data \
    $WORK/retristyle/retristyle-production.sif \
    ls -la /app/data

# Check that files exist on host
ls -la $WORK/colorist/data
```

### Issue: WandB not logging on cluster

**Solution:**
```bash
# Verify API key is set
echo $WANDB_API_KEY

# Use offline mode if network issues
export WANDB_MODE=offline

# Sync logs later
wandb sync $WORK/colorist/logs/wandb/offline-run-*
```

### Issue: Out of memory during training

**Solution:**
```bash
# Reduce batch size in training arguments
# Or request more memory in SLURM script:
#SBATCH --mem=512G

# Check memory usage
sstat -j <JOBID> --format=JobID,MaxRSS,MaxVMSize
```

### Issue: Training not resuming from checkpoint

**Solution:**
```bash
# Verify checkpoint exists
ls -la $WORK/colorist/checkpoints/reference_methods/adain/

# Check checkpoint is not corrupted
apptainer exec $WORK/retristyle/retristyle-production.sif \
    python -c "import torch; ckpt = torch.load('$WORK/colorist/checkpoints/reference_methods/adain/checkpoint_latest.pt'); print(ckpt.keys())"

# Ensure --resume_from_checkpoint flag is in training command
```

---

## Quick Reference Commands

### Docker Commands

```bash
# Build image
docker build --target production -t retristyle:production .

# Run container
docker run --rm --gpus all -v /data/local/colorist/data:/app/data retristyle:production python script.py

# Save image
docker save retristyle:production -o retristyle-production.tar

# Clean up
docker system prune -a
```

### Apptainer Commands

```bash
# Convert from Docker
apptainer build retristyle.sif docker-archive://retristyle.tar

# Run container
apptainer exec --bind /data:/app/data retristyle.sif python script.py

# Inspect image
apptainer inspect retristyle.sif

# Shell into container
apptainer shell --bind /data:/app/data retristyle.sif
```

### SLURM Commands

```bash
# Submit job
sbatch script.slurm

# Check queue
squeue -u $USER

# Cancel job
scancel <JOBID>

# View output
tail -f logs/job_<JOBID>.out

# Check quota
shownicerquota.pl
```

---

## Next Steps

After completing this guide, you should have:

1. ✅ Docker and Apptainer installed locally
2. ✅ Production Docker image built
3. ✅ Apptainer .sif file created
4. ✅ Local testing completed successfully
5. ✅ Required files downloaded
6. ✅ Container and data transferred to cluster
7. ✅ Training job running on HPC

**For more details:**
- See `README.Docker.md` for advanced Docker usage
- See `DEPLOYMENT_CHECKLIST.md` for quick deployment reference
- See `scripts/README.md` for available training scripts
- See project `README.md` for retristyle library usage

**Need help?**
- Check the troubleshooting section above
- Review NHR@FAU documentation: https://doc.nhr.fau.de/
- Contact xAILab Bamberg research group

---

**Document Version:** 1.0  
**Last Updated:** November 24, 2025  
**Maintained by:** xAILab Bamberg, University of Bamberg
