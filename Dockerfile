# Production Dockerfile for RetriStyle Deep Learning Project
# Optimized for HPC cluster deployment via Apptainer

# ============================================================================
# Base Stage - System dependencies and Python environment
# ============================================================================
ARG PYTORCH_VERSION=2.9.1
ARG CUDA_VERSION=12.8.0
ARG PYTHON_VERSION=3.13

# Note: Using -cudnn tag which includes cuDNN 9.x (exact version managed by NVIDIA)
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
# libmagickwand-dev: Required by medmnistc (Wand Python package for image corruptions/augmentations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libmagickwand-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ARG PYTHON_VERSION
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Accept Conda Terms of Service for required channels and create environment with Python
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda config --set channel_priority flexible && \
    conda config --set auto_activate_base false && \
    conda create -n retristyle python=${PYTHON_VERSION} -y && \
    conda clean -ya

# Activate environment by default
ENV CONDA_DEFAULT_ENV=retristyle \
    PATH=/opt/conda/envs/retristyle/bin:$PATH

SHELL ["/bin/bash", "-c"]

# ============================================================================
# Dependencies Stage - Install Python packages
# ============================================================================
FROM base AS dependencies

ARG PYTORCH_VERSION
ARG CUDA_VERSION

# Install PyTorch with CUDA 12.8 support (official release supports CUDA 12.8)
RUN pip install torch==${PYTORCH_VERSION} torchvision torchaudio

# Copy requirements file
WORKDIR /tmp/requirements
COPY requirements.txt ./

# Install all Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --upgrade pip "setuptools<81"
# RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

# ============================================================================
# Production Stage - Minimal runtime environment
# ============================================================================
FROM dependencies AS production

# Create working directory
WORKDIR /app

# Copy only necessary project files
COPY code/ /app/code/
COPY setup.py README.md __init__.py /app/

# Install package
RUN pip install .

# Create directories for mounted data, models, and results
RUN mkdir -p /app/data \
    /app/models \
    /app/results \
    /app/wandb

# Set defaults consumed by code/config/paths.py
ENV DATA_PATH=/app/data \
    MODEL_DIR=/app/models \
    WEIGHTS_DIR=/app/models/style_transfer \
    EMBEDDING_DIR=/app/data/embeddings \
    OUTPUT_PATH=/app/results

# Non-root user for security
# Note: Container runs as retristyle user, but on HPC the bind mounts
# will use the host user's UID, so file permissions work correctly
RUN useradd -m retristyle && \
    chown -R retristyle:retristyle /app
USER retristyle

# Default command
CMD ["python", "-c", "import code.retristyle; print('RetriStyle is ready!')"]
