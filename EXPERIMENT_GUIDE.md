# RetriStyle — Experiment Guide

This document explains the entire codebase and how to conduct all thesis experiments, from style transfer evaluation through ablation studies to cross-dataset extension.

---

## Table of Contents

1. [Codebase Overview](#1-codebase-overview)
2. [Prerequisites & Setup](#2-prerequisites--setup)
3. [Dataset Setup](#3-dataset-setup)
4. [Classifiers](#4-classifiers)
5. [Experiment 1: Style Transfer Method Comparison](#5-experiment-1-style-transfer-method-comparison)
6. [Experiment 2: Reference Baseline Training](#6-experiment-2-reference-baseline-training)
7. [Experiment 3: Geometric TTA Baseline](#7-experiment-3-geometric-tta-baseline)
8. [Experiment 4: RetriStyle Ablation Suite](#8-experiment-4-retristyle-ablation-suite)
9. [Experiment 5: Hybrid TTA (Geometric + Style Mixing)](#9-experiment-5-hybrid-tta-geometric--style-mixing)
10. [Experiment 6: Cross-Dataset Extension](#10-experiment-6-cross-dataset-extension)
11. [Evaluation & Reporting](#11-evaluation--reporting)
12. [HPC Workflow (NHR@FAU)](#12-hpc-workflow-nhrfau)
13. [Quick Reference](#13-quick-reference)

---

## 1. Codebase Overview

### Directory Structure

```
retristyle/
├── retristyle/                    # Core library
│   ├── infer_style_base.py        #   StyleID implementation (training-free)
│   ├── ensemble_utils.py          #   FOODS filter + soft voting
│   ├── diffusion/                 #   Style injection diffusion pipeline
│   └── retrieval/                 #   Reference DB + retrievers (random, DINO, metric, …)
│
├── experiments/                   # All experiment entry points
│   ├── train.py                   #   Classifier training (multi-augmentation)
│   ├── classifier_evaluation.py   #   Trained model evaluation
│   ├── evaluate_predictions.py    #   TTA prediction evaluation
│   ├── inference_tta.py           #   Unified TTA wrapper (backward-compatible)
│   ├── clip_classifier.py         #   CLIP zero-shot + DINOv2 linear probe
│   ├── style_transfer_evaluation.py  # Style transfer method comparison
│   ├── domain_shift_analysis.py   #   Domain gap quantification
│   │
│   ├── tta/                       #   Test-Time Adaptation modules
│   │   ├── run_inference.py       #     Main TTA inference (all methods)
│   │   ├── augmentation.py        #     Augmented view generation
│   │   ├── evaluation.py          #     Aggregation strategies (vanilla/zero/tpt/foods)
│   │   ├── hybrid.py              #     Hybrid geo+style TTA
│   │   ├── generate_augmented_images.py  # Pre-generate & cache augmented views
│   │   ├── extract_embeddings.py  #     DINO embedding extraction
│   │   ├── reference_db_setup.py  #     Reference database + retriever factory
│   │   ├── checkpoint.py          #     Per-sample checkpoint/resume
│   │   ├── collect_results.py     #     Merge per-run JSON predictions
│   │   └── constants.py           #     Shared enums, defaults, sweep values
│   │
│   ├── reference_methods/         #   Reference/baseline methods
│   │   ├── train_domainbed.py     #     DomainBed training (ERM, Mixup, RSC, …)
│   │   ├── train_sdg.py           #     Single Domain Generalization training
│   │   ├── train.py               #     Training with color transfer augmentation
│   │   ├── color_transfer_factory.py  # Factory for 21+ style transfer methods
│   │   ├── domainbed/             #     DomainBed implementations
│   │   ├── sdg/                   #     SDG implementations
│   │   ├── style_transfer/        #     Style transfer implementations
│   │   └── TTA/                   #     Reference TTA methods (TENT, …)
│   │
│   ├── data/                      #   Dataset factory + 30+ datasets
│   │   ├── _factory.py            #     create_dataset() routing
│   │   ├── _constants.py          #     Normalization stats, NUM_CLASSES, etc.
│   │   └── imagenet_variants.py   #     ImageNet-1k + R/A/C/P/Sketch/V2
│   │
│   ├── metrics/                   #   Color + content preservation metrics
│   ├── reporting/                 #   LaTeX tables + visualization
│   └── utils/                     #   Preprocessing, reproducibility, etc.
│
├── scripts/                       #   Bash scripts (run locally or generate HPC jobs)
│   ├── common.sh                  #     Shared config (paths, classifiers, datasets)
│   ├── style_transfer_eval.sh     #     [Local] Style transfer comparison
│   ├── train_reference_baselines.sh  # [Local] Train reference methods
│   ├── geometric_tta_eval.sh      #     [Local] Geometric TTA baseline
│   ├── ablation_tta.sh            #     [Local] RetriStyle ablation
│   ├── hybrid_tta.sh              #     [Local] Hybrid TTA
│   ├── extract_embeddings.sh      #     [Local] DINO embedding extraction
│   ├── generate_augmented_cache.sh#     [Local] Pre-generate augmented views
│   ├── domain_shift_analysis.sh   #     [Local] Domain shift quantification
│   ├── classifier_evaluation.sh   #     [Local] Evaluate trained classifiers
│   ├── evaluate_predictions.sh    #     [Local] Evaluate TTA predictions
│   ├── generate_tables_and_figures.sh  # [Local] LaTeX tables + figures
│   ├── generate_hpc_*.sh          #     HPC script generators (→ generated/)
│   └── generated/                 #     Auto-generated SLURM scripts (gitignored)
│
└── configs/                       #   Accelerate multi-GPU configs
```

### Key Design Patterns

- **Every experiment** can be run locally via `bash scripts/<experiment>.sh` with CLI flags.
- **HPC at scale**: Run `bash scripts/generate_hpc_<experiment>.sh` to create SLURM scripts in `scripts/generated/`, then submit via `submit_all.sh`.
- **Checkpointing**: TTA inference writes per-sample predictions to JSON. Jobs can resume from where they left off.
- **Caching**: DINO embeddings and augmented views can be pre-computed and cached to avoid redundant computation.
- **Modularity**: All Python entry points work via `python -m experiments.<module>`.

---

## 2. Prerequisites & Setup

### Local Machine

```bash
conda activate colorist
pip install -r requirements.txt
```

### HPC Cluster (NHR@FAU)

The code runs inside the Apptainer container `retristyle-production.sif`:

```bash
# Container should be at $WORK/retristyle/retristyle-production.sif
# Data at $WORK/retristyle/data/
# Models at $WORK/retristyle/models/
```

### Configuration

Edit `scripts/common.sh` to point at your local paths:

```bash
DATA_PATH="/data/local/retristyle/data"
WEIGHTS_DIR="/data/local/retristyle/models/style_transfer"
MODEL_DIR="/data/local/retristyle/models/training"
```

---

## 3. Dataset Setup

### Primary: ImageNet Suite

The primary experimental setup uses **ImageNet-1k** as the source/training domain and evaluates against its distribution-shifted variants.

| Split | Dataset | Images | Classes | Domain Shift |
|-------|---------|--------|---------|--------------|
| `train` | ImageNet-1k train | 1.28M | 1000 | Source |
| `val` | ImageNet-1k val | 50k | 1000 | In-distribution |
| `test_r` | **ImageNet-R** | 30k | 200 | Renditions (art, cartoons, graffiti, …) |
| `test_a` | ImageNet-A | 7.5k | 200 | Natural adversarial |
| `test_sketch` | ImageNet-Sketch | 50k | 1000 | Sketches |
| `test_v2` | ImageNet-V2 | 10k | 1000 | Reproduction shift |

**Why ImageNet-R for ablations?** ImageNet → ImageNet-R presents a style/texture-based domain shift that RetriStyle is designed to mitigate. It has 200 classes from ImageNet-1k, allowing us to use pretrained ImageNet classifiers directly without retraining. Using the full 30k images (not a subset) makes results more general and trustworthy. The ImageNet suite is ideal because we train on ImageNet-1k only once and evaluate against multiple test sets.

**Directory layout expected at `DATA_PATH`:**

```
<DATA_PATH>/imagenet/
├── imagenet1k/
│   ├── train/  (1000 class folders)
│   └── val/    (1000 class folders)
├── imagenet-r/  (200 class folders)
├── imagenet-a/  (200 class folders)
├── imagenet-sketch/  (1000 class folders)
└── imagenetv2-matched-frequency-format-val/  (1000 folders)
```

### Extension Datasets (add when time permits)

Uncomment in `scripts/common.sh`:

```bash
# Natural domain generalization
NATURAL_DATASETS=("pacs" "vlcs" "domainnet" "office_home" "terra_incognita")
# Medical
MEDICAL_DATASETS=("camelyon17wilds" "epistr" "peripheral_blood" "fitzpatrick17k" "retina")
```

---

## 4. Classifiers

We use two representatives per architecture family, all pretrained:

| Family | Model | Pretrained On | Fine-tune Needed? |
|--------|-------|---------------|-------------------|
| CNN | `resnet18` | ImageNet-1k | Head only (for ImageNet-R 200 classes) |
| CNN | `densenet121` | ImageNet-1k | Head only |
| ViT | `vit_base_patch16_224` | ImageNet-1k | Head only |
| ViT | `swin_base_patch4_window7_224` | ImageNet-1k | Head only |
| VLM | `ViT-B-16` (CLIP) | LAION/OpenAI | No — zero-shot via text prompts |
| FM | `dinov2_vitb14` (DINOv2) | LVD-142M | Linear probe on frozen features |

**CLIP and DINOv2 use `--weights_path pretrained`** — no training file needed.
**CNN/ViT models** need training first (Experiment 2), producing weights at:
`<MODEL_DIR>/<dataset>-<classifier>-<augmentation>-seed<SEED>.pth`

---

## 5. Experiment 1: Style Transfer Method Comparison

**Goal:** Find the best style transfer method for TTA. Evaluate all 17+ methods (artistic trained, artistic diffusion, photorealistic) using identical protocol.

**Protocol:** 20 content images from ImageNet-R × 40 style images from ImageNet-1k = 800 stylized images per method. Metrics: Wasserstein distance, histogram intersection, SSIM, LPIPS, edge similarity.

### Run Locally

```bash
# All methods
bash scripts/style_transfer_eval.sh

# Specific methods
bash scripts/style_transfer_eval.sh --methods styleid diffstyle adain
```

### Run on HPC

```bash
# Generate SLURM scripts (one per method)
bash scripts/generate_hpc_style_transfer_eval.sh

# Submit all
bash scripts/generated/style_transfer_eval/submit_all.sh
```

### Python Entry Point

```bash
python -m experiments.style_transfer_evaluation \
    --data_path /data --weights_dir /data/models/style_transfer \
    --output_dir ./results/style_transfer_eval \
    --content_dataset imagenet --content_split test_r \
    --style_dataset imagenet --style_split train \
    --n_content 20 --n_style 40 --seed 42
```

### Expected Output

`results/style_transfer_eval/all_methods_comparison.json` with per-method aggregated metrics. Ideally **StyleID** ranks among the best (high content preservation + good style transfer), since our TTA pipeline builds on it.

---

## 6. Experiment 2: Reference Baseline Training

**Goal:** Train all reference methods to establish baselines for comparison.

### Methods

- **A) Augmentation baselines** (5): none, color_jitter, rand_augment, trivial_augment, aug_mix
- **B) DomainBed** (6): ERM, IB_ERM, Mixup, RSC, SD, SelfReg
- **C) SDG** (1): MixStyle

All trained with 3 seeds × 4 trainable classifiers (CNN + ViT). CLIP and DINOv2 are pretrained.

### Run Locally

```bash
# Augmentation baseline
bash scripts/train_reference_baselines.sh --method augmentation \
    --classifier resnet18 --augmentation color_jitter --seed 71397589

# DomainBed
bash scripts/train_reference_baselines.sh --method domainbed \
    --classifier resnet18 --domainbed_algorithm ERM --seed 71397589

# SDG
bash scripts/train_reference_baselines.sh --method sdg \
    --classifier resnet18 --sdg_method MixStyle --seed 71397589
```

### Generate HPC Scripts

```bash
# Generates: 4 classifiers × (5 aug + 6 domainbed + 1 sdg) × 3 seeds = 144 scripts
bash scripts/generate_hpc_train_reference_baselines.sh
bash scripts/generated/reference_baselines/submit_all.sh
```

### Evaluate Trained Models

```bash
bash scripts/classifier_evaluation.sh --classifier resnet18 \
    --augmentation none --seed 71397589
```

---

## 7. Experiment 3: Geometric TTA Baseline

**Goal:** Establish geometric augmentation TTA performance (ZERO paper baseline) across all classifiers and evaluation strategies. This serves as the reference for style transfer TTA comparison.

Evaluates on ImageNet → ImageNet-R with `n_views=64` geometric augmentations.

### Run Locally

```bash
bash scripts/geometric_tta_eval.sh --classifier ViT-B-16 --eval_strategy zero
```

### Generate HPC Scripts

```bash
# 6 classifiers × 3 eval strategies × 3 seeds = 54 scripts
bash scripts/generate_hpc_geometric_tta_eval.sh
bash scripts/generated/geometric_tta/submit_all.sh
```

### What This Tells Us

- Which evaluation strategy (vanilla/zero/tpt) works best with geometric augmentations
- Per-sample success/failure patterns — which test images geometric TTA gets wrong
- Baseline accuracy across all classifier architectures

---

## 8. Experiment 4: RetriStyle Ablation Suite

**Goal:** Systematically optimize the RetriStyle TTA pipeline on ImageNet → ImageNet-R.

### Step 0: Pre-compute Embeddings

Required for DINO retrieval strategy:

```bash
bash scripts/extract_embeddings.sh --dataset imagenet --split train
# Or on HPC:
bash scripts/generate_hpc_extract_embeddings.sh
bash scripts/generated/extract_embeddings/submit_all.sh
```

### Ablation A: Retrieval Strategy

**Fix:** eval_strategy=zero, n_refs=16, classifier=ViT-B-16
**Sweep:** random, balanced_random, metric, balanced_metric, dino

```bash
# Local (single combination)
bash scripts/ablation_tta.sh --retrieval_strategy dino \
    --eval_strategy zero --n_refs 16 --classifier ViT-B-16

# HPC (all combinations across classifiers × seeds)
bash scripts/generate_hpc_ablation_tta.sh
bash scripts/generated/ablation/submit_all.sh
```

### Ablation B: Aggregation Strategy

**Fix:** best retrieval from A, n_refs=16
**Sweep:** vanilla, zero, tpt

```bash
bash scripts/ablation_tta.sh --eval_strategy tpt \
    --retrieval_strategy dino --n_refs 16
```

### Ablation C: Number of References (n_refs)

**Fix:** best retrieval + best eval from A/B
**Sweep:** 2, 4, 8, 16, 32, 64

```bash
bash scripts/ablation_tta.sh --n_refs 4 \
    --retrieval_strategy dino --eval_strategy zero
```

### Updating Best Settings

After each ablation, update defaults for subsequent ones:

```bash
BEST_RETRIEVAL=dino BEST_EVAL=zero BEST_N_REFS=16 \
    bash scripts/generate_hpc_ablation_tta.sh
```

### Optional: Pre-generate Augmented Image Cache

To avoid re-doing style transfer when evaluating multiple classifiers:

```bash
bash scripts/generate_augmented_cache.sh --n_refs 16 --retrieval_strategy dino

# HPC:
bash scripts/generate_hpc_augmented_cache.sh
bash scripts/generated/augmented_cache/submit_all.sh
```

Then pass `--augmented_views_dir <cache_root>` to `run_inference.py` to load pre-generated views.

---

## 9. Experiment 5: Hybrid TTA (Geometric + Style Mixing)

**Goal:** Determine whether combining geometric and style-transfer augmentations outperforms either alone. Compare against best geometric TTA setup.

### Mixing Ratios

| Ratio | Geometric Views | Style Views | Description |
|-------|----------------|-------------|-------------|
| (1.0, 0.0) | 64 | 0 | Pure geometric (= Experiment 3) |
| (0.75, 0.25) | 48 | 16 | Mostly geometric |
| (0.5, 0.5) | 32 | 32 | Half and half |
| (0.25, 0.75) | 16 | 48 | Mostly style |
| (0.0, 1.0) | 0 | 64 | Pure style (= Experiment 4) |

### Run Locally

```bash
bash scripts/hybrid_tta.sh --classifier ViT-B-16 --geo_frac 0.5
```

### Generate HPC Scripts

```bash
# 6 classifiers × 5 ratios × 3 seeds = 90 scripts
BEST_RETRIEVAL=dino BEST_EVAL=zero BEST_N_REFS=16 \
    bash scripts/generate_hpc_hybrid_tta.sh
bash scripts/generated/hybrid_tta/submit_all.sh
```

### Key Question This Answers

- Is style transfer TTA a **replacement** for geometric TTA? → Pure style beats pure geometric.
- Or is it an **extension**? → Mixed ratios (e.g., 50/50) beat both pure approaches.
- Per-sample analysis: which images does style TTA correctly classify that geometric TTA misses?

---

## 10. Experiment 6: Cross-Dataset Extension

After determining the best setup from Experiments 4-5, extend to additional test sets.

### Phase 1: Other ImageNet Variants

Evaluate best RetriStyle setup on all ImageNet test splits:

```bash
# Run ablation with different split
bash scripts/ablation_tta.sh --split test_a    # ImageNet-A
bash scripts/ablation_tta.sh --split test_sketch  # ImageNet-Sketch
bash scripts/ablation_tta.sh --split test_v2   # ImageNet-V2
```

### Phase 2: Natural Domain Generalization Datasets

Uncomment in `scripts/common.sh` and re-run:

```bash
# After uncommenting NATURAL_DATASETS
bash scripts/generate_hpc_train_reference_baselines.sh  # train references
bash scripts/generate_hpc_ablation_tta.sh               # run RetriStyle
```

### Phase 3: Medical Datasets

Same procedure with `MEDICAL_DATASETS`. Note: Medical datasets may need different normalization and class-count settings.

---

## 11. Evaluation & Reporting

### Collect & Merge Results

```bash
bash scripts/evaluate_predictions.sh --results_dir ./results
```

### Generate Tables & Figures

```bash
bash scripts/generate_tables_and_figures.sh --results_dir ./results
```

This produces:
- `results/tables/`: LaTeX tables (style transfer comparison, ablation results, hybrid TTA, cross-dataset)
- `results/figures/`: Publication-quality plots (bar charts, line plots, confusion matrices)

### Domain Shift Analysis

Visualize the nature of domain gaps to argue when style-transfer TTA helps:

```bash
bash scripts/domain_shift_analysis.sh --target_split test_r
bash scripts/domain_shift_analysis.sh --target_split test_a
bash scripts/domain_shift_analysis.sh --target_split test_sketch
```

---

## 12. HPC Workflow (NHR@FAU)

### General Pattern

```bash
# 1. Generate SLURM scripts
bash scripts/generate_hpc_<experiment>.sh

# 2. Review generated scripts
ls scripts/generated/<experiment>/

# 3. Submit all jobs
bash scripts/generated/<experiment>/submit_all.sh

# 4. Monitor
squeue -u $USER

# 5. After completion, collect results locally or on cluster
bash scripts/evaluate_predictions.sh --results_dir $WORK/retristyle/results
```

### Container Binding Pattern

All HPC scripts use the same Apptainer execution pattern:

```bash
apptainer exec --nv \
    --bind $DATA_PATH:/app/data \
    --bind $OUTPUT_PATH:/app/results \
    --bind $EMBEDDING_DIR:/app/embeddings \
    --bind $HF_MODELS_CACHE:/app/hf_models \
    --bind $TORCH_MODELS_CACHE:/app/torch_models \
    $CONTAINER \
    python -m experiments.<module> --data_path /app/data ...
```

### Job Timings

| Experiment | GPU | Approximate Time |
|-----------|-----|-----------------|
| Style transfer eval (1 method) | 1× A40 | 2-4h |
| Augmentation training | 1× A40 | 12-24h |
| Geometric TTA (full dataset) | 1× A40 | 6-12h |
| RetriStyle ablation (n_refs≤16) | 2× A40 | 12-24h |
| RetriStyle ablation (n_refs=64) | 4× A40 | 12-24h |
| Augmented cache generation | 2-4× A40 | 24-48h |

### Auto-Resubmission

All TTA scripts include timeout-based auto-resubmission:
```bash
timeout 23h apptainer exec ...
[[ $EXIT_CODE -eq 124 ]] && sbatch "${BASH_SOURCE[0]}"
```
Combined with per-sample checkpointing, jobs will resume where they left off.

---

## 13. Quick Reference

### Recommended Experiment Order

```
1. Extract embeddings          → scripts/extract_embeddings.sh
2. Style transfer comparison   → scripts/style_transfer_eval.sh
3. Domain shift analysis       → scripts/domain_shift_analysis.sh
4. Train reference baselines   → scripts/generate_hpc_train_reference_baselines.sh
5. Geometric TTA baseline      → scripts/generate_hpc_geometric_tta_eval.sh
6. RetriStyle ablation A/B/C   → scripts/generate_hpc_ablation_tta.sh
7. Pre-generate augmented views→ scripts/generate_hpc_augmented_cache.sh
8. Hybrid TTA mixing           → scripts/generate_hpc_hybrid_tta.sh
9. Cross-dataset extension     → modify common.sh, re-run generators
10. Evaluate & generate tables  → scripts/evaluate_predictions.sh
                                  scripts/generate_tables_and_figures.sh
```

### Python Entry Points

| Task | Command |
|------|---------|
| Train classifier | `python -m experiments.train` |
| Train DomainBed | `python -m experiments.reference_methods.train_domainbed` |
| Train SDG | `python -m experiments.reference_methods.train_sdg` |
| TTA inference | `accelerate launch -m experiments.tta.run_inference` |
| Hybrid TTA | `python -m experiments.tta.hybrid` |
| Style transfer eval | `python -m experiments.style_transfer_evaluation` |
| Domain shift | `python -m experiments.domain_shift_analysis` |
| Extract embeddings | `python -m experiments.tta.extract_embeddings` |
| Cache augmentations | `python -m experiments.tta.generate_augmented_images` |
| Evaluate predictions | `python -m experiments.evaluate_predictions` |
| Classifier eval | `python -m experiments.classifier_evaluation` |
| LaTeX tables | `python -m experiments.reporting.generate_latex_tables` |
| Visualizations | `python -m experiments.reporting.visualize_results` |

### Key Constants (experiments/tta/constants.py)

```python
ALL_SEEDS = [71397589, 133560673, 265017005]
DEFAULT_N_VIEWS = 64
ZERO_GAMMA = 0.3    # retain top 30% views
TPT_GAMMA = 0.1     # retain top 10% views
THESIS_CLASSIFIERS = {
    "cnn": ["resnet18", "densenet121"],
    "vit": ["vit_base_patch16_224", "swin_base_patch4_window7_224"],
    "vlm": ["ViT-B-16"],
    "fm":  ["dinov2_vitb14"],
}
THESIS_HYBRID_RATIOS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]
```
