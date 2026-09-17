# Original RetriStyle-TTA — Test-Time Adaptation Experiments setup
# Usable for STyle-TTA

This document covers how to **run TTA inference** locally (Docker) and on
the NHR@FAU HPC cluster, and describes the two main evaluation setups.

---

## Table of Contents

1. [Evaluation Setups](#evaluation-setups)
2. [Package Structure](#package-structure)
3. [Prerequisites](#prerequisites)
4. [Quick Local Test (Docker)](#quick-local-test)
5. [Full Local Sweep (Docker)](#full-local-sweep)
6. [Embedding Extraction](#embedding-extraction)
7. [HPC Deployment (NHR@FAU)](#hpc-deployment)
8. [CLI Reference](#cli-reference)
9. [TTA Methods](#tta-methods)
10. [Evaluation Strategies](#evaluation-strategies)
11. [Retrieval Strategies](#retrieval-strategies)
12. [Evaluating Predictions](#evaluating-predictions)

---

## Evaluation Setups

We evaluate TTA inference under **two setups**, both compared against
RetriStyle and TENT baselines. Every experiment is repeated across three
random seeds: **71397589**, **133560673**, **265017005**.

### Setup A — Baseline Classifier + Augmentation TTA

A classifier trained **without any augmentation** (`none`) is evaluated
at test time with each augmentation-based TTA method:

| Classifier training | TTA method at test time | Eval strategies |
|---------------------|------------------------|-----------------|
| `none` | `geometric` | vanilla, zero, foods |
| `none` | `gray_scale` | vanilla, zero, foods |
| `none` | `color_jitter` | vanilla, zero, foods |
| `none` | `auto_augment` | vanilla, zero, foods |
| `none` | `rand_augment` | vanilla, zero, foods |
| `none` | `trivial_augment` | vanilla, zero, foods |
| `none` | `aug_mix` | vanilla, zero, foods |
| `none` | `random_resized_crop` | vanilla, zero, foods |
| `none` | `random_flip` | vanilla, zero, foods |
| `none` | `random_erasing` | vanilla, zero, foods |
| `none` | `targeted_augment` | vanilla, zero, foods |
| `none` | `oracle` | vanilla, zero, foods |

**Purpose:** Measure how much each augmentation TTA helps an unaugmented
classifier — "free" improvement at test time without any training changes.

### Setup B — Augmentation-Trained Classifier + Matching TTA

A classifier trained **with augmentation X** is evaluated at test time
with the *same* augmentation X as TTA:

| Classifier training | TTA method at test time | Eval strategies |
|---------------------|------------------------|-----------------|
| `gray_scale` | `gray_scale` | vanilla, zero, foods |
| `color_jitter` | `color_jitter` | vanilla, zero, foods |
| `auto_augment` | `auto_augment` | vanilla, zero, foods |
| `rand_augment` | `rand_augment` | vanilla, zero, foods |
| `trivial_augment` | `trivial_augment` | vanilla, zero, foods |
| `aug_mix` | `aug_mix` | vanilla, zero, foods |
| `random_resized_crop` | `random_resized_crop` | vanilla, zero, foods |
| `random_flip` | `random_flip` | vanilla, zero, foods |
| `random_erasing` | `random_erasing` | vanilla, zero, foods |
| `targeted_augment` | `targeted_augment` | vanilla, zero, foods |

**Purpose:** Measure whether combining a training augmentation with
matching TTA at test time yields further gains on top of training
augmentations alone.

### Baselines

Both setups are compared against:

| Method | Classifier | Eval strategy | Notes |
|--------|-----------|---------------|-------|
| **RetriStyle** | `none` | vanilla, zero, foods | Style-transfer TTA via diffusion |
| **TENT** | `none` | vanilla | BatchNorm affine adaptation |

### Weights Naming Convention

Classifier weights follow the pattern:

```
<dataset>-<classifier>-<augmentation>-seed<seed>.pth
```

Examples:
- `epistr-densenet121-none-seed71397589.pth` — no training augmentation
- `epistr-densenet121-color_jitter-seed71397589.pth` — trained with color jitter

### Running the Sweeps

**Local (Docker) — all setups at once:**

```bash
bash scripts/inference_tta/run_tta_experiments.sh
```

**Virgo server (Docker):**

```bash
bash scripts/inference_tta/run_tta_experiments_geom_virgo.sh
```

**HPC (SLURM) — generate + submit per-dataset per-seed scripts:**

```bash
# Setup A example: baseline classifier + geometric TTA
bash scripts/inference_tta/create_inference_tta_scripts.sh \
    geometric zero random none

# Setup B example: color_jitter-trained classifier + color_jitter TTA
bash scripts/inference_tta/create_inference_tta_scripts.sh \
    color_jitter zero random color_jitter

# RetriStyle baseline
bash scripts/inference_tta/create_inference_tta_scripts.sh \
    style_tta zero dino none

# Then submit
bash scripts/inference_tta/geometric_zero_random_none/submit_all.sh
```

---

## Package Structure

The TTA inference pipeline lives in **`experiments/tta/`** — a modular
package that replaces the original monolithic `experiments/inference_tta.py`
(which now delegates to the package).

```
experiments/tta/
├── __init__.py              # Package init — re-exports public API
├── constants.py             # Shared constants (seed, ZERO params, method lists)
├── augmentation.py          # View generation: geometric, style-transfer, distributed
├── evaluation.py            # Eval strategies: vanilla, ZERO, FOODS, TENT
├── checkpoint.py            # Prediction persistence / resume (atomic JSON writes)
├── reference_db_setup.py    # Reference database & retriever factory
├── extract_embeddings.py    # Standalone timm embedding extraction & caching
└── run_inference.py         # Main orchestrator — CLI entry-point
```

| Module | Responsibility |
|--------|---------------|
| `constants.py` | `DEFAULT_SEED`, `ZERO_N_VIEWS`, `ZERO_GAMMA`, available method/strategy lists |
| `augmentation.py` | Produces `(V, 3, H, W)` views for each TTA method. `augment_views()` (single-GPU) / `augment_views_distributed()` (multi-GPU via Accelerate) |
| `evaluation.py` | `eval_vanilla()` — average softmax; `eval_zero()` — entropy-filter + majority vote; `eval_foods()` — OOD-filtered weighted ensemble; `eval_tent()` — BatchNorm affine adaptation |
| `checkpoint.py` | JSON predictions with atomic writes; auto-resume on SLURM timeout |
| `reference_db_setup.py` | `build_reference_db()` — lazy-loading training set; `build_retriever()` — factory for all 5 strategies; `materialise_images()` — for FOODS |
| `extract_embeddings.py` | Pre-computes & caches timm embeddings (default: DINOv3 ViT-B/16) to `<dir>/<dataset>/<model_tag>/<split>.pt` |
| `run_inference.py` | Wires everything: dataset → embeddings → retrieval → augmentation → evaluation → metrics → save |

**Backward compatibility:** `experiments/inference_tta.py` still exists as a
thin wrapper that imports and calls `experiments.tta.run_inference.main()`, so
all existing scripts continue to work.

---

## Prerequisites

### Docker Image

Build the Docker image from the project root:

```bash
docker build -t style_tta:latest .
```

### Required Data

- **Datasets** — MedMNIST, Camelyon17, EpiSTR, etc. in a local directory
  (e.g. `./data`).
- **Classifier weights** — Pre-trained `.pth` files
  (e.g. `./models/pathmnist-densenet121-none-seed265017005.pth`).

### Python Packages (non-Docker)

If running outside Docker, install the project and its dependencies:

```bash
pip install -e .               # style_tta package
pip install timm faiss-cpu     # DINOv3 embeddings + FAISS search
```

---

## Quick Local Test

Run a single test to verify the pipeline works:

```bash
# Augmentation TTA (geometric + ZERO)
docker run --rm --gpus 1 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/results:/app/results \
    -e PYTHONPATH=/app -e HF_HOME=/app/hf_models -e TORCH_HOME=/app/torch_models \
    style_tta:production \
    python -m code.experiments.tta.run_inference \
        --dataset pathmnist \
        --data_path /app/data \
        --classifier densenet121 \
        --weights_path /app/models/pathmnist-densenet121-none-seed265017005.pth \
        --tta_method geometric \
        --eval_strategy zero \
        --n_views 16 \
        --output_path /app/results
```

### Quick Test with RetriStyle (1 reference)

```bash
docker run --rm --gpus 1 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/results:/app/results \
    -e PYTHONPATH=/app -e HF_HOME=/app/hf_models -e TORCH_HOME=/app/torch_models \
    style_tta:production \
    python -m code.experiments.tta.run_inference \
        --dataset pathmnist \
        --data_path /app/data \
        --classifier densenet121 \
        --weights_path /app/models/pathmnist-densenet121-none-seed265017005.pth \
        --tta_method style_tta \
        --eval_strategy zero \
        --retrieval_strategy random \
        --n_refs 1 \
        --output_path /app/results
```

### Quick Test with TENT

```bash
docker run --rm --gpus 1 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/results:/app/results \
    -e PYTHONPATH=/app -e HF_HOME=/app/hf_models -e TORCH_HOME=/app/torch_models \
    style_tta:production \
    python -m code.experiments.tta.run_inference \
        --dataset pathmnist \
        --data_path /app/data \
        --classifier densenet121 \
        --weights_path /app/models/pathmnist-densenet121-none-seed265017005.pth \
        --tta_method tent \
        --eval_strategy vanilla \
        --output_path /app/results
```

---

## Full Local Sweep

The sweep scripts run all setups (A, B, RetriStyle, TENT) across all
configured datasets and all three seeds automatically:

```bash
# Edit DATASETS array in the script to select datasets, then:
bash scripts/inference_tta/run_tta_experiments.sh
```

For a single dataset with RetriStyle + DINO retrieval (64 references):

```bash
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/embeddings:/app/embeddings \
    -e PYTHONPATH=/app -e HF_HOME=/app/hf_models -e TORCH_HOME=/app/torch_models \
    style_tta:production \
    python -m code.experiments.tta.run_inference \
        --dataset pathmnist \
        --data_path /app/data \
        --classifier densenet121 \
        --weights_path /app/models/pathmnist-densenet121-none-seed265017005.pth \
        --tta_method style_tta \
        --eval_strategy zero \
        --retrieval_strategy dino \
        --n_refs 64 \
        --embedding_dir /app/embeddings \
        --output_path /app/results
```

### Multi-GPU with Accelerate

When using >1 GPU, pass the Accelerate config:

```bash
docker run --rm --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/embeddings:/app/embeddings \
    -e PYTHONPATH=/app -e HF_HOME=/app/hf_models -e TORCH_HOME=/app/torch_models \
    style_tta:production \
    accelerate launch --config_file /app/configs/gpu_04.yaml \
        -m experiments.tta.run_inference \
        --dataset pathmnist \
        --data_path /app/data \
        --classifier densenet121 \
        --weights_path /app/models/pathmnist-densenet121-none-seed265017005.pth \
        --tta_method style_tta \
        --eval_strategy zero \
        --retrieval_strategy dino \
        --n_refs 64 \
        --embedding_dir /app/embeddings \
        --output_path /app/results
```

The `configs/gpu_XX.yaml` files are numbered by GPU count (01–08).

---

## Embedding Extraction

When using `--retrieval_strategy dino`, the pipeline needs timm embeddings
for both the training set (retrieval pool) and the test set (query).

**Automatic:** If `--embedding_dir` is passed, `run_inference.py`
automatically extracts and caches embeddings before the inference loop
starts. If the files already exist, extraction is skipped.

**Manual / Independent:** You can also pre-extract embeddings as a
standalone step (useful for running once across multiple experiments):

```bash
# Inside Docker
docker run --rm --gpus 1 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/embeddings:/app/embeddings \
    style_tta:latest \
    python -m code.experiments.tta.extract_embeddings \
        --dataset pathmnist \
        --data_path /app/data \
        --output_dir /app/embeddings \
        --splits train test \
        --model_name vit_base_patch16_dinov3.lvd1689m

# Without Docker
python -m code.experiments.tta.extract_embeddings \
    --dataset pathmnist \
    --data_path ./data \
    --output_dir ./embeddings \
    --splits train test
```

Cache layout:
```
embeddings/
└── pathmnist/
    └── vit_base_patch16_dinov3_lvd1689m/
        ├── train.pt    # (N_train, 768) L2-normalised
        └── test.pt     # (N_test, 768)  L2-normalised
```

---

## HPC Deployment

### Script Generator

The script `scripts/inference_tta/create_inference_tta_scripts.sh` generates
per-dataset **per-seed** SLURM scripts + a `submit_all.sh` launcher.
A separate script is created for each `(dataset, seed)` combination.

```bash
# Generate scripts for RetriStyle + ZERO + DINO retrieval (baseline classifier)
bash scripts/inference_tta/create_inference_tta_scripts.sh \
    style_tta zero dino none "gpu:a100:1" "a100"

# Generate scripts for color_jitter + ZERO (aug-trained classifier — Setup B)
bash scripts/inference_tta/create_inference_tta_scripts.sh \
    color_jitter zero random color_jitter
```

**Arguments:**

| Position | Name | Values |
|----------|------|--------|
| 1 | `tta_method` | Any from [TTA Methods](#tta-methods) |
| 2 | `eval_strategy` | `vanilla`, `zero`, `foods` |
| 3 | `retrieval_strategy` | `random`, `balanced_random`, `metric`, `balanced_metric`, `dino` |
| 4 | `augmentation` | Training augmentation tag (e.g. `none`, `color_jitter`, `rand_augment`) |
| 5 | `gpu_config` (optional) | SLURM `--gres` value (default: `gpu:a100:1`) |
| 6 | `partition` (optional) | SLURM partition (default: `a100`) |

This creates:

```
scripts/inference_tta/style_tta_zero_dino_none/
├── submit_all.sh                              # Submit all jobs
├── infer_hpc_pathmnist_seed71397589.sh        # Per-dataset per-seed
├── infer_hpc_pathmnist_seed133560673.sh
├── infer_hpc_pathmnist_seed265017005.sh
├── infer_hpc_dermamnist_seed71397589.sh
├── ...
└── infer_hpc_retina_seed265017005.sh
```

### Submit Jobs

```bash
bash scripts/inference_tta/style_tta_zero_dino_none/submit_all.sh
```

### Key HPC Features

- **24h wall time** with 23h timeout — if the job times out, it
  automatically resubmits itself (`sbatch "${BASH_SOURCE[0]}"`)
- **Checkpoint/resume** — predictions are saved every 10 samples;
  restarted jobs skip already-processed samples
- **Embedding caching** — `--embedding_dir` ensures embeddings are
  extracted once and reused across restarts
- **Apptainer** — the Docker image is converted to `.sif` and executed
  with `--nv` for GPU passthrough

### Convert Docker → Apptainer

```bash
# On HPC login node
apptainer build colorist-production.sif docker-archive://style_tta_latest.tar
```

---

## CLI Reference

### `experiments.tta.run_inference`

Main entry-point. Can be launched directly or via `experiments.inference_tta`
(backward-compatible wrapper).

```
python -m code.experiments.tta.run_inference [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | *required* | Dataset name (e.g. `pathmnist`, `dermamnist`) |
| `--data_path` | str | *required* | Path to dataset root |
| `--classifier` | str | *required* | timm classifier model name |
| `--weights_path` | str | *required* | Path to classifier `.pth` weights |
| `--tta_method` | str | *required* | TTA method (see [TTA Methods](#tta-methods)) |
| `--eval_strategy` | str | `zero` | `vanilla`, `zero`, `foods` |
| `--retrieval_strategy` | str | `random` | `random`, `balanced_random`, `metric`, `balanced_metric`, `dino` |
| `--n_refs` | int | `64` | Number of style references per test image |
| `--n_views` | int | `64` | Number of stochastic views for augmentation TTA |
| `--embedding_dir` | str | `None` | Cache directory for DINOv3 embeddings |
| `--embedding_model` | str | `vit_base_patch16_dinov3.lvd1689m` | timm model for embeddings |
| `--color_method` | str | `None` | Color-transfer method (required for `color_tta`) |
| `--color_method_weights` | str | `None` | Weights for color-transfer model |
| `--tent_lr` | float | `1e-3` | TENT learning rate |
| `--tent_steps` | int | `1` | TENT adaptation steps |
| `--zero_gamma` | float | `0.3` | ZERO: fraction of most-confident views |
| `--foods_tau` | float | `2.0` | FOODS OOD threshold |
| `--seed` | int | `265017005` | Random seed |
| `--input_size` | int | `224` | Classifier input resolution |
| `--native_size` | int | `512` | Style-transfer native resolution |
| `--batch_size` | int | `128` | Batch size (FOODS centroids, etc.) |
| `--num_workers` | int | `4` | DataLoader workers |
| `--split` | str | `test` | Evaluation split |
| `--output_path` | str | `./results` | Output directory for results + predictions |

### `experiments.tta.extract_embeddings`

Standalone embedding extraction.

```
python -m code.experiments.tta.extract_embeddings [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | *required* | Dataset name |
| `--data_path` | str | *required* | Dataset root |
| `--output_dir` | str | `./embeddings` | Where to save `.pt` files |
| `--model_name` | str | `vit_base_patch16_dinov3.lvd1689m` | timm model |
| `--splits` | str+ | `train test` | Splits to extract |
| `--input_size` | int | `224` | Resize before extraction |
| `--batch_size` | int | `64` | Batch size |
| `--device` | str | `cuda` | Device |
| `--force` | flag | — | Re-extract even if cached |

---

## TTA Methods

### Augmentation-based TTA

These methods generate `N` stochastic views of a single test image by
applying a random augmentation. The original image is always included as
one of the views. No retrieval or reference database is needed.

| Method | Description |
|--------|-------------|
| `geometric` | Random resized crop + horizontal flip (ZERO paper default) |
| `gray_scale` | Random grayscale conversion |
| `color_jitter` | Random brightness, contrast, saturation, hue shifts |
| `auto_augment` | AutoAugment (ImageNet policy) |
| `rand_augment` | RandAugment (2 ops, magnitude 9) |
| `trivial_augment` | TrivialAugmentWide |
| `aug_mix` | AugMix (3 chains, depth 2–3) |
| `random_resized_crop` | Random resized crop only |
| `random_flip` | Random horizontal flip only |
| `random_erasing` | Random erasing (CutOut-style) |
| `targeted_augment` | Domain-specific augmentation from MedMNISTC |
| `oracle` | Randomly picks a different augmentation from the pool for each view |

### Retrieval-based TTA

These methods retrieve `n_refs` style references from the training set and
use style transfer to create augmented views. The retrieval pool always
spans the entire training set.

| Method | Description |
|--------|-------------|
| `adain_tta` | AdaIN style transfer |
| `color_tta` | Classical color-transfer methods (21 variants) |
| `style_tta` | Diffusion-based style transfer via StyleID |

### Other

| Method | Description |
|--------|-------------|
| `tent` | Test-time Entropy minimisation — adapts BatchNorm affine parameters. Single-pass, no views. Always uses `vanilla` eval strategy. |

---

## Evaluation Strategies

| Strategy | Paper | Description | Requires |
|----------|-------|-------------|----------|
| **vanilla** | — | Average softmax over all augmented views | views |
| **zero** | Farina et al. (NeurIPS 2024) | Entropy-filter top γ confident views, majority-vote at zero temperature | views |
| **foods** | — | Weighted ensemble using training-set centroid distances, OOD filtering past τ | views + training images |

All three strategies can be combined with any augmentation or retrieval TTA
method. TENT always uses `vanilla` (it does its own adaptation).

---

## Retrieval Strategies

Only relevant when `--tta_method` is `adain_tta`, `color_tta`, or
`style_tta`. Ignored for augmentation-based methods and TENT.

| Strategy | Description | Requires |
|----------|-------------|----------|
| **random** | Uniform random selection from training set | — |
| **balanced_random** | Class-balanced random selection | labels |
| **metric** | SSIM or Mutual Information scoring | image loading |
| **balanced_metric** | MMR-diversified metric retrieval | image loading |
| **dino** | DINOv3 ViT-B/16 embedding similarity via FAISS | pre-extracted embeddings |

---

## Evaluating Predictions

After inference completes, predictions and results are stored as per-dataset
JSON files:

- `results/tta_inference/predictions/<dataset>.json` — per-sample predictions
  (keyed by experiment configuration)
- `results/tta_inference/results/<dataset>.json` — final metrics
  (keyed by experiment configuration)

To re-evaluate predictions:

```bash
# Single dataset
python -m code.experiments.evaluate_predictions \
    --predictions_path ./results/tta_inference/predictions/epistr.json

# All datasets
python -m code.experiments.evaluate_predictions \
    --predictions_dir ./results/tta_inference/predictions/
```

---

## Example: Full Experiment Sweep (Local Docker)

```bash
#!/bin/bash
# Full local experiment sweep — adjust paths and GPU count as needed

IMAGE="style_tta:production"
DATA="$(pwd)/data"
MODELS="$(pwd)/models"
RESULTS="$(pwd)/results"
EMBEDDINGS="$(pwd)/embeddings"
DATASET="pathmnist"
CLASSIFIER="densenet121"
SEED=265017005

# Weights paths
W_NONE="$DATASET-$CLASSIFIER-none-seed$SEED.pth"
W_CJ="$DATASET-$CLASSIFIER-color_jitter-seed$SEED.pth"

DOCKER="docker run --rm --gpus 1
    -v $DATA:/app/data:ro -v $MODELS:/app/models:ro
    -v $RESULTS:/app/results -v $EMBEDDINGS:/app/embeddings
    -e PYTHONPATH=/app -e CUDA_VISIBLE_DEVICES=0
    -e HF_HOME=/app/hf_models -e TORCH_HOME=/app/torch_models
    $IMAGE"

# 1. Extract embeddings (once, reused by all dino experiments)
$DOCKER python -m code.experiments.tta.extract_embeddings \
    --dataset "$DATASET" --data_path /app/data \
    --output_dir /app/embeddings --splits train test

# 2. Setup A: Baseline classifier + geometric TTA + ZERO
$DOCKER python -m code.experiments.tta.run_inference \
    --dataset "$DATASET" --data_path /app/data \
    --classifier "$CLASSIFIER" --weights_path "/app/models/$W_NONE" \
    --tta_method geometric --eval_strategy zero \
    --n_views 64 --seed $SEED --output_path /app/results

# 3. Setup A: Baseline classifier + oracle TTA + ZERO
$DOCKER python -m code.experiments.tta.run_inference \
    --dataset "$DATASET" --data_path /app/data \
    --classifier "$CLASSIFIER" --weights_path "/app/models/$W_NONE" \
    --tta_method oracle --eval_strategy zero \
    --n_views 64 --seed $SEED --output_path /app/results

# 4. Setup B: color_jitter-trained + color_jitter TTA + ZERO
$DOCKER python -m code.experiments.tta.run_inference \
    --dataset "$DATASET" --data_path /app/data \
    --classifier "$CLASSIFIER" --weights_path "/app/models/$W_CJ" \
    --tta_method color_jitter --eval_strategy zero \
    --n_views 64 --seed $SEED --output_path /app/results

# 5. RetriStyle + ZERO + DINO retrieval (64 refs)
$DOCKER python -m code.experiments.tta.run_inference \
    --dataset "$DATASET" --data_path /app/data \
    --classifier "$CLASSIFIER" --weights_path "/app/models/$W_NONE" \
    --tta_method style_tta --eval_strategy zero \
    --retrieval_strategy dino --n_refs 64 \
    --embedding_dir /app/embeddings --seed $SEED \
    --output_path /app/results

# 6. TENT baseline
$DOCKER python -m code.experiments.tta.run_inference \
    --dataset "$DATASET" --data_path /app/data \
    --classifier "$CLASSIFIER" --weights_path "/app/models/$W_NONE" \
    --tta_method tent --eval_strategy vanilla \
    --seed $SEED --output_path /app/results

# 7. Evaluate all predictions
docker run --rm -v "$RESULTS":/app/results "$IMAGE" \
    python -m code.experiments.evaluate_predictions \
        --predictions_dir /app/results/tta_inference/predictions/
```
