# STyle-TTA

STyle-TTA is a research framework for **structure-aware retrieval and diffusion-based test-time adaptation (TTA)**. It evaluates whether retrieved style references and style-transfer transformations can improve image classification under distribution and domain shifts.

The project combines:

- retrieval of reference images using random, metric, or DINO-based strategies;
- style-transfer and diffusion methods for generating test-time views;
- TTA baselines such as geometric augmentation and TENT;
- evaluation strategies including vanilla averaging, ZERO, TPT, and FOODS;
- experiments across ImageNet variants and medical and natural-domain datasets.

## Repository Layout

```text
retristyle/
├── retristyle/          Core STyle-TTA library
├── experiments/         Training, TTA, evaluation, metrics, and reporting
├── scripts/             Local experiment runners and HPC job generators
├── config/              Shared project configuration and constants
├── data/                Datasets, embeddings, and cached artifacts
├── checkpoints/         Model checkpoints
├── results/             Predictions and experiment outputs
├── figures/             Generated figures
├── requirements.txt     Reproducible Python dependencies
└── Dockerfile           CUDA-enabled production image
```

## Requirements

- Linux or another environment supported by PyTorch
- Python 3.10 or newer
- A CUDA-capable GPU for most style-transfer and TTA experiments
- Dataset files and classifier checkpoints

The tested environment described by the repository uses Python 3.13, PyTorch 2.9.1, and CUDA 12.8. CPU execution is suitable for lightweight development and inspection, but full experiments can require substantial GPU memory and storage.

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

For a containerized environment:

```bash
docker build --target production -t retristyle:production .
```

See [GETTING_STARTED.md](GETTING_STARTED.md) for Docker, NVIDIA Container Toolkit, and Apptainer setup details.

## Quick Start

Run the TTA entry point directly with a dataset and classifier checkpoint:

```bash
python -m experiments.tta.run_inference \
    --dataset pathmnist \
    --data_path ./data \
    --classifier densenet121 \
    --weights_path ./checkpoints/model.pth \
    --tta_method geometric \
    --eval_strategy zero \
    --n_views 16 \
    --output_path ./results
```

For STyle-TTA retrieval-based TTA:

```bash
python -m experiments.tta.run_inference \
    --dataset pathmnist \
    --data_path ./data \
    --classifier densenet121 \
    --weights_path ./checkpoints/model.pth \
    --tta_method retristyle \
    --eval_strategy zero \
    --retrieval_strategy random \
    --n_refs 1 \
    --output_path ./results
```

The available TTA methods, evaluation strategies, and retrieval strategies are documented in the module help and in [README_TTA.md](README_TTA.md):

```bash
python -m experiments.tta.run_inference --help
```

## Running Experiments

Experiment wrappers live in `scripts/` and use paths configured in `scripts/common.sh`.

```bash
# Style-transfer method comparison
bash scripts/style_transfer_eval.sh

# Train reference classifiers
bash scripts/train_reference_baselines.sh

# Geometric TTA baseline
bash scripts/geometric_tta_eval.sh

# STyle-TTA ablations
bash scripts/ablation_tta.sh

# Hybrid geometric + style TTA
bash scripts/hybrid_tta.sh

# Evaluate saved predictions and generate reports
bash scripts/evaluate_predictions.sh
bash scripts/generate_tables_and_figures.sh
```

For the complete experiment matrix, dataset layouts, classifier details, and HPC commands, read [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md). The thesis-specific implementation context is in [MASTER_THESIS_IMPLEMENTATION_PLAN.md](MASTER_THESIS_IMPLEMENTATION_PLAN.md).

## Data and Checkpoints

By default, the shell scripts expect data and model paths configured in `scripts/common.sh`. Override them for a local setup:

```bash
export DATA_PATH="$PWD/data"
export OUTPUT_PATH="$PWD/results"
export MODEL_DIR="$PWD/data/models"
export WEIGHTS_DIR="$PWD/data/weights"
```

Classifier checkpoints generally follow this naming pattern:

```text
<dataset>-<classifier>-<augmentation>-seed<seed>.pth
```

Do not commit large datasets, model weights, generated caches, or experiment logs to source control.

## HPC Deployment

The project supports Docker-to-Apptainer deployment for SLURM-based clusters:

```bash
./prepare-hpc.sh --target production --apptainer
```

HPC job-generation scripts are available under `scripts/`, including generators for style-transfer evaluation, TTA, embedding extraction, and baseline training.

## Documentation

- [GETTING_STARTED.md](GETTING_STARTED.md): environment, Docker, and HPC setup
- [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md): full experiment and dataset guide
- [README_TTA.md](README_TTA.md): TTA setups, CLI usage, and evaluation strategies
- [MASTER_THESIS_IMPLEMENTATION_PLAN.md](MASTER_THESIS_IMPLEMENTATION_PLAN.md): implementation and thesis plan

## Citation

This repository contains the implementation for the STyle-TTA research project. Add the project publication citation here once the associated paper or thesis metadata is finalized.

## License

The package metadata identifies this project as MIT licensed. Add the repository's full license text to a `LICENSE` file before distributing releases.