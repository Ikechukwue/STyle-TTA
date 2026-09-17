# STyle-TTA

STyle-TTA is a research framework for **structure-aware retrieval and diffusion-based test-time adaptation (TTA)**. It evaluates whether retrieved style references and style-transfer transformations can improve image classification under distribution and domain shifts.

## Provenance and Thesis Scope

This repository contains the implementation developed for a master's thesis on
STyle-TTA. It is based on the RetriStyle project code originally developed by
**Sebastian Dörrich** and made available for adaptation in this thesis.

The thesis work adapts and extends that codebase for retrieval-driven,
training-free style-transfer TTA. In particular, the thesis-specific work
includes the experiment orchestration, retrieval configurations, TTA
evaluation, classifier comparisons, ablations, and reporting used for the
results described in the thesis. The repository also retains inherited and
general-purpose research components that are not necessarily used in every
thesis experiment; their presence does not imply that they are thesis
contributions.

The project combines:

- retrieval of reference images using random, metric, or DINO-based strategies;
- style-transfer and diffusion methods for generating test-time views;
- TTA baselines such as geometric augmentation;
- evaluation strategies including vanilla averaging, ZERO, TPT;
- experiments across ImageNet variants and medical and natural-domain datasets.

## Repository Layout

```text
retristyle/
├── code/
│   ├── config/          Paths, constants, and GPU configuration
│   ├── experiments/     Training, inference, TTA, evaluation, and reporting
│   ├── retristyle/      Retrieval, style-transfer, and ensemble library code
│   └── scripts/         Local and HPC experiment wrappers
├── data/                Local datasets, embeddings, and augmented caches
├── models/              Local model weights, checkpoints, and style-transfer weights
├── results/             Local predictions, statistics, figures, and reports
├── requirements.txt     Experiment environment dependencies
├── setup.py             Editable package installation metadata
├── Dockerfile           CUDA-enabled production image
└── prepare-hpc.sh       Docker-to-Apptainer deployment helper
```

Large datasets, model weights, checkpoints, generated outputs, and experiment
results are intentionally excluded from Git. Configure their locations with
`DATA_PATH`, `MODEL_DIR`, `WEIGHTS_DIR`, `EMBEDDING_DIR`, and `OUTPUT_PATH`, or
create local `data/`, `models/`, and `results/` directories at the repository
root. Do not commit machine-specific symlinks to external storage paths.

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

The container copies the canonical `code/` source tree into `/app/code`.
Dataset and model directories should be mounted into the container rather than
committed to the repository.

## Quick Start

Run the TTA entry point directly with a dataset and classifier checkpoint:

```bash
python -m code.experiments.tta.run_inference \
    --dataset pathmnist \
    --data_path ./data \
    --classifier densenet121 \
    --weights_path ./models/model.pth \
    --tta_method geometric \
    --eval_strategy zero \
    --n_views 16 \
    --output_path ./results
```

For STyle-TTA retrieval-based TTA:

```bash
python -m code.experiments.tta.run_inference \
    --dataset pathmnist \
    --data_path ./data \
    --classifier densenet121 \
    --weights_path ./models/model.pth \
    --tta_method retristyle \
    --eval_strategy zero \
    --retrieval_strategy random \
    --n_refs 1 \
    --output_path ./results
```

The available TTA methods, evaluation strategies, and retrieval strategies are documented in the module help and in [README_TTA.md](README_TTA.md):

```bash
python -m code.experiments.tta.run_inference --help
```

## Running Experiments

Experiment wrappers live in `code/scripts/` and use paths configured in
`code/scripts/common.sh`. Run them from the repository root:

```bash
# Style-transfer method comparison
bash code/scripts/style_transfer_eval.sh

# Train reference classifiers
bash code/scripts/train_reference_baselines.sh

# Geometric TTA baseline
bash code/scripts/geometric_tta_eval.sh

# STyle-TTA ablations
bash code/scripts/ablation_tta.sh

# Hybrid geometric + style TTA
bash code/scripts/hybrid_tta.sh

# Evaluate saved predictions and generate reports
bash code/scripts/evaluate_predictions.sh
bash code/scripts/generate_tables_and_figures.sh
```

The TTA entry points and evaluation strategies are documented in
[README_TTA.md](README_TTA.md). The source tree also contains reference
methods and exploratory reporting code inherited from the underlying research
project; use the thesis experiment commands and configurations as the supported
submission workflow.

## Data and Checkpoints

By default, the shell scripts expect data and model paths configured in
`code/scripts/common.sh`. Override them for a local setup:

```bash
export DATA_PATH="$PWD/data"
export OUTPUT_PATH="$PWD/results"
export MODEL_DIR="$PWD/models"
export WEIGHTS_DIR="$PWD/models/style_transfer"
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

HPC job-generation scripts are available under `code/scripts/`, including
generators for style-transfer evaluation, TTA, embedding extraction, and
baseline training.

## Documentation

- [README_TTA.md](README_TTA.md): TTA setups, CLI usage, and evaluation strategies
- [setup.py](setup.py): editable package metadata and runtime dependencies
- [requirements.txt](requirements.txt): experiment environment dependencies and version constraints

## Citation

This repository contains the implementation for the STyle-TTA research project. Add the project publication citation here once the associated paper or thesis metadata is finalized.

## License

The package metadata identifies this project as MIT licensed. Add the repository's full license text to a `LICENSE` file before distributing releases.
