# Master Thesis Implementation Plan
# Style Transfer as Test-Time Augmentation for Domain Generalization
# Comprehensive Architectural Plan & Implementation Blueprint

---

## Table of Contents

1. [Current Architecture Summary](#1-current-architecture-summary)
2. [Codebase Cleanup & Refactoring](#2-codebase-cleanup--refactoring)
3. [Dataset Strategy & Dataloader Setup](#3-dataset-strategy--dataloader-setup)
4. [Domain Shift Analysis Framework](#4-domain-shift-analysis-framework)
5. [Style Transfer Method Evaluation](#5-style-transfer-method-evaluation)
6. [Classifier Architecture Setup](#6-classifier-architecture-setup)
7. [Reference Baselines (DG & Traditional TTA)](#7-reference-baselines-dg--traditional-tta)
8. [Generative TTA Pipeline & Ablation Suite](#8-generative-tta-pipeline--ablation-suite)
9. [Hybrid TTA (Geometric + Style Transfer)](#9-hybrid-tta-geometric--style-transfer)
10. [Extension to Full Dataset Battery](#10-extension-to-full-dataset-battery)
11. [Infrastructure, Checkpointing & HPC Deployment](#11-infrastructure-checkpointing--hpc-deployment)
12. [Results Aggregation & LaTeX Reporting](#12-results-aggregation--latex-reporting)
13. [Execution Order & Dependencies](#13-execution-order--dependencies)

---

## 1. Current Architecture Summary

### 1.1 Project Structure Overview

The codebase is organized into three main packages:

```
retristyle/                          # Core library
├── infer_style_base.py              # StyleID (CVPR 2024) — flagship method
├── ensemble_utils.py                # FOODS OOD filter + soft vote aggregation
├── diffusion/
│   ├── style_injection.py           # StyleInjectionDiffusion (extended StyleID)
│   └── kv_cache.py                  # KV-cache for 50% FLOP reduction
└── retrieval/
    ├── base.py                      # BaseRetriever ABC
    ├── reference_db.py              # ReferenceDatabase (lazy loading)
    ├── random_retriever.py          # Uniform random sampling
    ├── balanced_random_retriever.py # Class-balanced random
    ├── metric_retriever.py          # SSIM/MI on gradient maps
    ├── balanced_metric_retriever.py # MMR re-ranking
    ├── dino_retriever.py            # DINOv3 + FAISS
    └── build_retrieval_db.py        # Offline DB construction

experiments/                         # Experiment scripts
├── train.py                         # Main training (augmentation-aware, checkpointed)
├── train_orig.py                    # Legacy duplicate — CAN BE DEPRECATED
├── inference_tta.py                 # Thin wrapper → tta/run_inference.py
├── inference_reporting.py           # Timing/throughput benchmarks
├── classifier_evaluation.py         # Model loading, metrics (Acc, BalAcc, AUC, ECE)
├── evaluate_predictions.py          # Offline metric computation from JSON
├── data/                            # Dataset loaders (20+ datasets)
│   ├── _factory.py                  # create_dataset() dispatcher
│   ├── _constants.py                # Normalization, class counts, splits
│   ├── imagenet_variants.py         # ImageNet-1k + A/R/C/P/Sketch/V2
│   ├── pacs.py / vlcs.py           # Leave-one-domain-out DG benchmarks
│   └── fitzpatrick.py / ddi.py     # Dermatology (skin-tone shifts)
├── tta/                             # TTA inference engine
│   ├── run_inference.py             # Main loop with per-sample checkpointing
│   ├── augmentation.py              # View generation (geometric + style)
│   ├── evaluation.py                # Aggregation: vanilla, ZERO, TPT, FOODS, TENT
│   ├── checkpoint.py                # Atomic JSON save/resume
│   ├── reference_db_setup.py        # DB construction + retriever factory
│   ├── extract_embeddings.py        # DINO embedding extraction + caching
│   ├── collect_results.py           # Merge per-run JSONs
│   └── constants.py                 # Seeds, gamma values, method lists
├── reference_methods/
│   ├── color_transfer_factory.py    # 21 style transfer methods unified
│   ├── color_transfer.py            # Style transfer quality evaluation
│   ├── train_domainbed.py           # DomainBed regularization (ERM, Mixup, RSC, SD, SelfReg, IB_ERM)
│   ├── train_sdg.py                 # Single-DG methods (JiGen, etc.)
│   ├── domainbed/                   # DomainBed algorithm implementations
│   ├── sdg/                         # SDG algorithm implementations
│   ├── style_transfer/              # 21 method wrappers (training-required + training-free)
│   └── TTA/                         # Geometric TTA, TENT baselines
├── metrics/
│   ├── color_metrics.py             # Wasserstein, histogram, FID, ArtFID, Gatys style loss
│   └── content_metrics.py           # SSIM, LPIPS, edge similarity, depth consistency
└── utils/                           # Preprocessing, reproducibility, training helpers
```

### 1.2 Data Pipeline

- `create_dataset(name, path, split, transform, **kwargs)` is the single entry point
- All DG datasets (PACS, VLCS, DomainNet, etc.) support leave-one-domain-out via `train_domain` kwarg
- ImageNet supports 7 OOD test splits: `test_a`, `test_r`, `test_c`, `test_p`, `test_sketch`, `test_v2`
- Subset sampling is natively supported (`use_subset`, `subset_size`, `subset_seed`)

### 1.3 TTA Inference Pipeline

The existing TTA pipeline (`experiments/tta/run_inference.py`) already supports:
- **TTA methods**: geometric, color_tta (21 reference methods), retristyle (StyleID), tent
- **Evaluation strategies**: vanilla (avg), ZERO (entropy-filter + majority vote), TPT (confidence-filter), FOODS (OOD-weighted)
- **Retrieval**: random, balanced_random, metric (SSIM/MI), balanced_metric (MMR), dino (FAISS)
- **Checkpointing**: Per-sample atomic JSON writes, full resume on restart
- **Multi-GPU**: Accelerate-based distribution of style transfers

### 1.4 Existing HPC Infrastructure

- Dockerfile with CUDA 12.8 + PyTorch 2.9.1 + Apptainer conversion
- SLURM script generators (`create_training_scripts_baseline.sh`, `create_retristyle_inference_scripts.sh`)
- Auto-restart on 24h timeout (exit code 124 → sbatch self)
- Proxy configuration for NHR@FAU nodes

### 1.5 Available Resources

**On disk at `/data/local/retristyle/models/style_transfer/`:**
21 pretrained style transfer model weights (adaattn.pth, adain.pth, artflow.pth, cast.pth, efdm.pth, iecontrast.pth, mast.pth, sanet.pth, styleformer.pth, stytr2.pth, aespanet.pth, deeppreset.tar, diffstyle.pt, diffuseit.pt, modflows.pt, photonas.pth.tar, wct2.pt, etc.)

**No datasets currently at `/data/local/retristyle/data/`** — all must be downloaded/prepared.

---

## 2. Codebase Cleanup & Refactoring

### 2.1 Files to Deprecate

| File | Reason | Action |
|------|--------|--------|
| `experiments/train_orig.py` | Duplicate of `train.py` | Mark deprecated, keep as backup |
| `experiments/utils/preprocessing_backup.py` | Backup copy | Remove |
| `scripts/training/create_training_scripts_colorist_NEEDSREDO.sh` | Marked as needing redo | Remove |
| `scripts/training/create_training_scripts_colorist_orig.sh` | Superseded | Remove |

### 2.2 New Directory Structure

Add the following directories to accommodate the thesis experiments:

```
experiments/
├── thesis/                              # NEW: Thesis-specific experiment scripts
│   ├── __init__.py
│   ├── domain_shift_analysis.py         # NEW: Domain shift characterization (Phase 4)
│   ├── style_transfer_evaluation.py     # NEW: Comparative ST evaluation (Phase 5)
│   ├── hybrid_tta.py                    # NEW: Geometric + Style Transfer mixing
│   └── generate_augmented_images.py     # NEW: Generate & cache augmented images to disk

scripts/
├── thesis/                              # NEW: Thesis experiment script generators
│   ├── create_dataset_download.sh       # NEW: Download/verify all datasets
│   ├── create_domain_shift_analysis.sh  # NEW: Domain shift analysis scripts
│   ├── create_style_transfer_eval.sh    # NEW: Style transfer comparison scripts
│   ├── create_classifier_training.sh    # NEW: Classifier fine-tuning scripts
│   ├── create_reference_baselines.sh    # NEW: DG reference method training scripts
│   ├── create_geometric_tta_eval.sh     # NEW: Geometric TTA evaluation scripts
│   ├── create_ablation_retrieval.sh     # NEW: Retrieval method ablation scripts
│   ├── create_ablation_aggregation.sh   # NEW: Aggregation strategy ablation scripts
│   ├── create_ablation_nrefs.sh         # NEW: N-refs sweep ablation scripts
│   ├── create_hybrid_tta_eval.sh        # NEW: Hybrid TTA mixing ratio scripts
│   ├── create_full_eval_extension.sh    # NEW: Extension to all datasets
│   └── create_latex_tables_thesis.sh    # NEW: LaTeX table generation
```

### 2.3 Configuration Constants Update

Extend `experiments/data/_constants.py` to ensure all thesis datasets have proper normalization stats. Currently ImageNet variants all share `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` which is correct since they share the class space.

### 2.4 Update `experiments/tta/constants.py`

Add new constants:

```python
# Thesis-specific constants
THESIS_PRIMARY_DATASET = "imagenet"  # Primary ablation dataset
THESIS_PRIMARY_TRAIN_SPLIT = "train"
THESIS_PRIMARY_TEST_SPLIT = "test_r"  # ImageNet-R as primary OOD test

THESIS_CLASSIFIERS = {
    "cnn": ["resnet18", "densenet121"],
    "vit": ["vit_base_patch16_224", "swin_base_patch4_window7_224"],
    "vlm": ["ViT-B-16"],  # CLIP
    "fm": ["vit_base_patch16_dinov3.lvd1689m"],  # DINOv3
}

THESIS_N_REFS_SWEEP = [2, 4, 8, 16, 32, 64]
THESIS_HYBRID_RATIOS = [
    (1.0, 0.0),   # pure geometric
    (0.75, 0.25), # 3/4 geo, 1/4 style
    (0.5, 0.5),   # half and half
    (0.25, 0.75), # 1/4 geo, 3/4 style
    (0.0, 1.0),   # pure style
]

THESIS_SEEDS = [71397589, 133560673, 265017005]
```

---

## 3. Dataset Strategy & Dataloader Setup

### 3.1 Dataset Decision: ImageNet Suite as Primary

**Recommendation: Use the full ImageNet-R (30,000 images) as the primary OOD test set, with ImageNet-1k as the training/source domain.**

Rationale:
1. **No training required**: All chosen classifiers (ResNet-18, DenseNet-121, ViT-B/16, SWIN-B, CLIP, DINOv3) come pretrained on ImageNet-1k. We only need to fine-tune heads for some architectures.
2. **Rich domain shift**: ImageNet-R contains "renditions" (art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, video game renditions) — exactly the kind of style/texture shift our method addresses.
3. **Multiple test sets from one training**: Train once on ImageNet-1k, then evaluate against ImageNet-R, ImageNet-A, ImageNet-C, ImageNet-Sketch, ImageNet-V2 — each representing a different type of domain shift.
4. **Community standard**: Results are directly comparable to published ZERO, TPT, and other TTA papers.

**Important note on ImageNet-R/A**: These datasets contain only 200 of the 1000 ImageNet classes. The implementation agent must:
- Filter the ImageNet-1k training set to include **only the 200 overlapping classes** when building the retrieval database for ImageNet-R/A experiments.
- Remap class indices from the ImageNet-R/A folder structure (which uses the original ImageNet wnid folders) to a contiguous 0-199 range, or keep the original 1000-class structure and mask the softmax output to the 200 relevant classes during evaluation.
- The existing `imagenet_variants.py` loader already handles this (it uses `ImageFolder` which auto-discovers classes from the directory structure).

### 3.2 Dataset Download & Verification

**The implementation agent MUST check `/data/local/retristyle/data/` for each dataset before proceeding and provide download instructions.**

Create `scripts/thesis/create_dataset_download.sh`:

```bash
#!/bin/bash
# Dataset download/verification script for thesis experiments
# Checks /data/local/retristyle/data/ for each required dataset
# Downloads those that are freely available, provides manual instructions for others

DATA_DIR="/data/local/retristyle/data"

# === Priority 1: ImageNet Suite ===
# ImageNet-1k: MANUAL - requires academic account at image-net.org
# Check: $DATA_DIR/imagenet/imagenet1k/train/ and val/
# ImageNet-R: AUTO-DOWNLOADABLE
# wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar -P $DATA_DIR/imagenet/
# ImageNet-A: AUTO-DOWNLOADABLE
# wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar -P $DATA_DIR/imagenet/
# ImageNet-Sketch: Requires Google Drive or HuggingFace download
# ImageNet-V2: AUTO-DOWNLOADABLE from HuggingFace

# === Priority 2: DG Benchmarks ===
# PACS: Download via DomainBed or Google Drive
# VLCS: Download via DomainBed

# === Priority 3: Medical ===
# Fitzpatrick17k: Available from GitHub
# DDI: Requires clinical data agreement
```

### 3.3 Ablation Subset DataLoader

For rapid local testing, use the existing `use_subset` mechanism:

```python
# Quick ablation: 1000 images from ImageNet-R
test_set = create_dataset(
    "imagenet", data_path, split="test_r",
    transform=transform,
    use_subset=True, subset_size=1000, subset_seed=42
)
```

No new code needed — the subsetting is already built into every dataset class.

### 3.4 Leave-One-Domain-Out Protocol (PACS/VLCS)

Already fully implemented. The `train_domain` kwarg controls which domain to train on:

```python
# PACS: Leave art_painting out
ds = create_dataset("pacs", path, "train", transform=tfm, train_domain="art_painting")
# trains on cartoon+photo+sketch, tests on art_painting

# VLCS: Leave CALTECH out
ds = create_dataset("vlcs", path, "train", transform=tfm, train_domain="CALTECH")
```

To iterate all domains, the script generator must loop over:
- PACS: `['art_painting', 'cartoon', 'photo', 'sketch']`
- VLCS: `['CALTECH', 'LABELME', 'PASCAL', 'SUN']`

### 3.5 ImageNet-R Class Filtering for Retrieval

**Critical implementation detail**: When building the reference database for TTA on ImageNet-R, we must restrict the retrieval DB to only the 200 ImageNet-R classes. The agent must add a `class_filter` parameter to `build_reference_db()` or create a dataset variant that filters classes:

```python
# New: experiments/data/_utils.py — add class filtering utility
def filter_dataset_by_classes(dataset, target_classes):
    """Filter a dataset to only include samples from target_classes."""
    indices = [i for i in range(len(dataset)) if dataset[i][1] in target_classes]
    return Subset(dataset, indices)
```

Alternatively, define a mapping of ImageNet-R wnid folders → class indices and expose it from `imagenet_variants.py`.

---

## 4. Domain Shift Analysis Framework

### 4.1 Motivation

Before running expensive TTA experiments, we should theoretically/quantitatively characterize the domain shifts to argue **why** style transfer TTA should help. This strengthens the thesis significantly.

### 4.2 Proposed Analyses

Create `experiments/thesis/domain_shift_analysis.py`:

#### 4.2.1 Statistical Domain Distance Metrics

Compute between source (ImageNet-1k) and each target (ImageNet-R, ImageNet-A, etc.):

1. **Fréchet Distance (FID-like)** on raw pixel statistics — measures overall distribution shift
2. **Color Distribution Distance**: Wasserstein distance on per-channel color histograms (already in `color_metrics.py` as `compute_wasserstein_distance`)
3. **Texture/Shape Statistics**:
   - **Gram Matrix Distance** (Gatys style loss, already in `color_metrics.py` as `compute_gatys_style_loss`): High Gram distance = large texture/style shift
   - **Edge Similarity** (SSIM on Sobel edges, already in `content_metrics.py`): High edge similarity = structure preserved, shift is in texture
4. **Feature-space Domain Distance**: Use DINOv3 embeddings, compute Maximum Mean Discrepancy (MMD) between source and target domains

#### 4.2.2 Texture vs. Shape Bias Analysis

For each domain pair:
1. Compute average **Gram matrix distance** (texture shift magnitude)
2. Compute average **edge/depth preservation score** (content preservation)
3. Plot Gram distance vs. edge preservation in 2D — domain pairs in the "high texture shift, preserved structure" quadrant are ideal for style transfer TTA

#### 4.2.3 Per-Class Domain Shift Heterogeneity

Some classes may shift more than others. Compute per-class:
- Color histogram Wasserstein distance
- LPIPS distance between random pairs
- Gram matrix distance

This helps predict which classes benefit from style transfer augmentation.

#### 4.2.4 Visualization: t-SNE / UMAP Embedding Plots

- Extract DINOv3 embeddings for source and target sets (using `experiments/tta/extract_embeddings.py`)
- Apply t-SNE/UMAP to visualize domain gap
- Color by class and by domain to show: (a) how separated the domains are, (b) whether class structure is preserved

### 4.3 Implementation

The analysis script takes sampled image pairs and computes all metrics:

```python
# experiments/thesis/domain_shift_analysis.py
# Inputs: source_dataset (ImageNet-1k val), target_dataset (ImageNet-R)
# Outputs: JSON with all domain distance metrics + visualization plots
# Re-uses existing metrics from experiments/metrics/
```

### 4.4 Key Argument for the Thesis

If results show:
- **ImageNet → ImageNet-R**: High Gram distance, preserved edge structure → "style shift" → style transfer TTA is well-suited
- **ImageNet → ImageNet-A**: Low Gram distance, different edge structure → "adversarial/semantic shift" → geometric TTA may be better
- **ImageNet → ImageNet-C**: Moderate Gram distance, degraded edges → "corruption shift" → domain-specific augmentation needed

This provides the theoretical grounding for why style transfer TTA works on certain domain shifts but not others.

---

## 5. Style Transfer Method Evaluation

### 5.1 Objective

Find the best style transfer method for TTA by comparing all 21(+) implemented methods.

### 5.2 Protocol (following existing `color_transfer.py`)

- **Content images**: 20 random images from ImageNet-R (representing OOD test samples)
- **Style images**: 40 random images from ImageNet-1k (representing training domain)
- **Total stylized images**: 20 × 40 = 800 per method
- **All methods frozen** (use pretrained weights from `/data/local/retristyle/models/style_transfer/`)

### 5.3 Methods to Evaluate

Categorized into three groups:

**Artistic (training-required)**:
adain, adaattn, aespanet, artflow, cast, efdm, iecontrast, mast, sanet, styleformer, stytr2

**Artistic Diffusion (training-free)**:
styleid, stylessp, instantstyle, diffstyle, diffuseit

**Photorealistic**:
modflows, wct2, deeppreset, photonas

### 5.4 Metrics (all already implemented)

**Style Transfer Quality** (from `experiments/metrics/color_metrics.py`):
- Wasserstein color distance (stylized vs. style reference)
- Histogram distance
- Color moment distance
- FID (between stylized and style domain)
- ArtFID
- Gatys style loss

**Content Preservation** (from `experiments/metrics/content_metrics.py`):
- Luminance SSIM (stylized vs. content)
- LPIPS distance
- Edge similarity (structural preservation)
- Depth consistency

### 5.5 Script

Adapt existing `experiments/reference_methods/color_transfer.py`:

```python
# experiments/thesis/style_transfer_evaluation.py
# For each method:
#   1. Load method via color_transfer_factory.create_color_transfer_method()
#   2. Generate 800 stylized images (20 content × 40 style)
#   3. Compute all metrics
#   4. Save results JSON + sample visualizations
#   5. Produce comparison table
```

### 5.6 Expected Outcome

Rank all methods by a composite quality score (style fidelity × content preservation). StyleID should rank highly due to its diffusion-based attention injection mechanism.

---

## 6. Classifier Architecture Setup

### 6.1 Architecture Selection

| Category | Model | timm Name | Pretrained | Head Fine-tune? |
|----------|-------|-----------|------------|-----------------|
| CNN | ResNet-18 | `resnet18` | ImageNet-1k | No (for ImageNet) / Yes (for PACS, etc.) |
| CNN | DenseNet-121 | `densenet121` | ImageNet-1k | No / Yes |
| ViT | ViT-B/16 | `vit_base_patch16_224` | ImageNet-1k | No / Yes |
| ViT | SWIN-B | `swin_base_patch4_window7_224` | ImageNet-1k | No / Yes |
| VLM | CLIP ViT-B/16 | `ViT-B-16` (open_clip) | CLIP | Yes (linear probe) |
| FM | DINOv3 ViT-B/16 | `vit_base_patch16_dinov3.lvd1689m` | DINOv3 | Yes (linear probe) |

### 6.2 ImageNet Pretrained Models

For ImageNet-suite experiments, **timm pretrained models can be used directly** — no training needed:

```python
import timm
model = timm.create_model("resnet18", pretrained=True, num_classes=1000)
```

For the 200-class ImageNet-R evaluation:
- **Option A**: Use the full 1000-class model, evaluate only on the 200 relevant classes (mask others to -∞ before softmax). This is the standard approach.
- **Option B**: Fine-tune a 200-class head. Less standard but may yield better absolute numbers.

**Recommendation**: Option A (standard evaluation protocol, comparable to published results).

### 6.3 CLIP Integration

CLIP requires special handling since it's not a standard timm model. The implementation agent must:

1. Use `open_clip` library to load CLIP ViT-B/16
2. For classification: use CLIP's zero-shot capability (text prompts per class) or train a linear probe on the visual features
3. For TTA: augmented views go through CLIP's visual encoder, predictions aggregated normally

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
# Zero-shot: encode class names as text, compute cosine similarity
# Linear probe: freeze CLIP visual encoder, train linear head
```

### 6.4 DINOv3 Integration

DINOv3 is already integrated for retrieval. For classification:
1. Use `timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True)`
2. Train a linear classification head on the frozen features

### 6.5 Fine-tuning Script for Non-ImageNet Datasets

For PACS, VLCS, and medical datasets, create classifier fine-tuning scripts:

```python
# Use existing experiments/train.py with:
#   --classifier resnet18  (or densenet121, vit_base_patch16_224, etc.)
#   --dataset pacs --train_domain photo
#   --augmentations none  (baseline: no augmentation)
#   --epochs 50 --early_stopping 10
#   --lr 1e-3 --batch_size 64
```

The existing `train.py` already supports all of this via timm.create_model with pretrained=True and custom num_classes.

---

## 7. Reference Baselines (DG & Traditional TTA)

### 7.1 Baseline Training Matrix

For each dataset × classifier × seed × method combination:

#### 7.1.1 Data Augmentation Baselines (via `experiments/train.py`)

Methods: `none`, `color_jitter`, `rand_augment`, `trivial_augment`, `aug_mix`, `auto_augment`, `gray_scale`, `random_resized_crop`, `random_flip`, `random_erasing`, `targeted_augment`

#### 7.1.2 DomainBed Methods (via `experiments/reference_methods/train_domainbed.py`)

Methods: `ERM`, `Mixup`, `RSC`, `SD`, `SelfReg`, `IB_ERM`

#### 7.1.3 SDG Methods (via `experiments/reference_methods/train_sdg.py`)

Methods: `JiGen` (and others as available)

### 7.2 Priority Schedule

**Round 1 (ImageNet — no training needed)**:
- Use pretrained classifiers directly on ImageNet-1k → ImageNet-R/A/C/Sketch/V2
- Only need TTA inference, no training

**Round 2 (ImageNet fine-tuned DG baselines — if needed)**:
- Train DomainBed methods on ImageNet-1k with domain augmentation
- This is expensive (ImageNet training) — only do if directly comparing to published results

**Round 3 (PACS/VLCS — small datasets)**:
- Full leave-one-domain-out protocol
- All classifiers × all DG methods × 3 seeds
- 4 domains × 6 DG methods × 6 classifiers × 3 seeds = 432 training runs per dataset

**Round 4 (Medical — Fitzpatrick, DDI)**:
- Skin-tone based domain split
- All classifiers × all methods × 3 seeds

### 7.3 Traditional (Geometric) TTA Evaluation

**This is a critical baseline**: run geometric TTA with the ZERO protocol to establish what standard TTA achieves.

```bash
# For each classifier pretrained on ImageNet-1k:
python -m experiments.inference_tta \
    --dataset imagenet --data_path /data --split test_r \
    --classifier resnet18 --weights_path PRETRAINED \
    --tta_method geometric --eval_strategy zero \
    --n_views 64 --seed 265017005
```

Run for all classifiers × all eval strategies (vanilla, zero, tpt) × 3 seeds.

### 7.4 TENT Baseline

```bash
python -m experiments.inference_tta \
    --dataset imagenet --data_path /data --split test_r \
    --classifier resnet18 --weights_path PRETRAINED \
    --tta_method tent --eval_strategy vanilla \
    --seed 265017005
```

### 7.5 Per-Sample Analysis

**Critical for thesis argumentation**: Save per-sample predictions from geometric TTA to later compare with style TTA. The existing checkpoint system already does this — every prediction is stored with `sample_idx`, `y_true`, and `y_pred` (full probability vector).

Create analysis script to identify:
- Samples correctly classified by geometric TTA
- Samples incorrectly classified by geometric TTA
- Later compare with style TTA to identify complementary strengths

---

## 8. Generative TTA Pipeline & Ablation Suite

### 8.1 Experimental Design

All ablations use ImageNet-1k → ImageNet-R as the primary setup. The pipeline follows a sequential narrowing approach — each experiment's best result feeds into the next.

### 8.2 Ablation 1: Retrieval Method

**Goal**: Which strategy best selects style references from ImageNet-1k?

| Retrieval | Description |
|-----------|-------------|
| `random` | Uniform random from training set |
| `balanced_random` | Class-balanced random |
| `dino` | DINOv3 embedding nearest-neighbor (FAISS) |
| `metric` (SSIM) | Structural similarity on gradient maps |
| `metric` (MI) | Mutual information on gradient maps |
| `balanced_metric` | MMR class-balanced metric |

**Fixed settings**: CLIP ViT-B/16 classifier, ZERO aggregation, 64 references, best style transfer method (StyleID).

```bash
# Template for each retrieval × seed:
python -m experiments.inference_tta \
    --dataset imagenet --data_path /data --split test_r \
    --classifier ViT-B-16 --weights_path CLIP_PRETRAINED \
    --tta_method retristyle --eval_strategy zero \
    --retrieval_strategy {random|dino|metric|balanced_random|balanced_metric} \
    --metric_type {ssim|mi}  # only for metric/balanced_metric
    --n_refs 64 --seed {seed}
```

### 8.3 Ablation 2: Aggregation Strategy

**Goal**: How to best aggregate predictions from multiple stylized views?

| Strategy | Description | Reference |
|----------|-------------|-----------|
| `vanilla` | Average softmax | Standard |
| `zero` | Entropy-filter + zero-temp majority vote | Farina et al., NeurIPS 2024 |
| `tpt` | Bottom-10% entropy filter + avg | Shu et al., NeurIPS 2022 |

**Fixed settings**: Best retrieval from Ablation 1, CLIP ViT-B/16, 64 refs, StyleID.

### 8.4 Ablation 3: Number of References (N-refs)

**Goal**: How many augmented views are needed?

| N-refs | Views | Note |
|--------|-------|------|
| 2 | 2 (1 original + 1 stylized) | Minimal |
| 4 | 4 | |
| 8 | 8 | |
| 16 | 16 | |
| 32 | 32 | |
| 64 | 64 | ZERO default |

**Fixed settings**: Best retrieval, best aggregation, CLIP ViT-B/16, StyleID.

### 8.5 Augmented Image Caching

**Critical optimization**: Store augmented images to disk so that different classifiers can reuse them without regenerating.

Create `experiments/thesis/generate_augmented_images.py`:

```python
# For each test sample in ImageNet-R:
#   1. Retrieve N references from ImageNet-1k training set
#   2. Generate N stylized views using StyleID
#   3. Save views to: /data/local/retristyle/augmented_cache/
#       imagenet_r/{retrieval}_{n_refs}/sample_{idx}/view_{v}.pt
#   4. Checkpointed: skip samples already on disk

# Later, for any classifier:
#   Load cached views, normalize, classify, aggregate
```

This avoids the O(classifiers × test_samples × n_refs) style transfer cost, reducing it to O(test_samples × n_refs) once.

### 8.6 Extension to All Classifiers

After determining the best setup (retrieval + aggregation + n_refs) from CLIP experiments, reuse cached augmented images to evaluate all classifiers:

```bash
for classifier in resnet18 densenet121 vit_base_patch16_224 swin_base_patch4_window7_224; do
    python -m experiments.inference_tta \
        --dataset imagenet --data_path /data --split test_r \
        --classifier $classifier --weights_path PRETRAINED \
        --tta_method retristyle --eval_strategy {best} \
        --retrieval_strategy {best} --n_refs {best_n} \
        --augmented_cache /data/local/retristyle/augmented_cache/ \
        --seed {seed}
done
```

The implementation agent must add `--augmented_cache` support to `run_inference.py`.

---

## 9. Hybrid TTA (Geometric + Style Transfer)

### 9.1 Motivation

If style transfer TTA doesn't universally outperform geometric TTA, combining both may be optimal. The hybrid approach generates some views via geometric augmentation and others via style transfer.

### 9.2 Mixing Ratios

For a total of N=64 views:

| Ratio | Geometric Views | Style Transfer Views |
|-------|----------------|---------------------|
| 1.0/0.0 | 64 | 0 |
| 0.75/0.25 | 48 | 16 |
| 0.5/0.5 | 32 | 32 |
| 0.25/0.75 | 16 | 48 |
| 0.0/1.0 | 0 | 64 |

### 9.3 Implementation

Create `experiments/thesis/hybrid_tta.py`:

```python
def augment_hybrid_views(test_img, n_total, geo_ratio, retriever, style_fn, ...):
    n_geo = int(n_total * geo_ratio)
    n_style = n_total - n_geo
    
    geo_views = augment_views(test_img, "geometric", n_geo, ...)
    style_views = augment_views(test_img, "retristyle", n_style, retriever=retriever, ...)
    
    return torch.cat([geo_views, style_views], dim=0)
```

Alternatively, extend `run_inference.py` to accept `--hybrid_geo_ratio` parameter.

### 9.4 Analysis

Compare per-sample predictions between:
- Pure geometric TTA (64 views)
- Pure style TTA (64 views)
- Best hybrid ratio

Identify samples where:
- Geometric wins, style fails → geometric augmentation essential for these
- Style wins, geometric fails → style captures domain shift better
- Both win → both work, any is fine
- Both fail → neither TTA helps, fundamentally hard samples

This analysis directly addresses the thesis question of complementarity.

---

## 10. Extension to Full Dataset Battery

### 10.1 Priority Order

After establishing the methodology on ImageNet-R, extend to other datasets in this order:

#### Priority 1: Other ImageNet Test Sets (no retraining needed)
- ImageNet-A (adversarial natural examples)
- ImageNet-C (corruptions — 15 types × 5 severities)
- ImageNet-Sketch
- ImageNet-V2

Use the same pretrained classifiers and the same best TTA configuration.

#### Priority 2: Natural DG Benchmarks (need training per domain)
- **PACS**: 4 domains (photo, art, cartoon, sketch) — very relevant since "sketch" and "art" are style shifts
- **VLCS**: 4 domains (CALTECH, LABELME, PASCAL, SUN) — photographic domain shift

For each: full leave-one-domain-out protocol with all classifiers × reference DG methods + our TTA.

#### Priority 3: Medical Datasets
- **Fitzpatrick17k**: Skin-tone based domain shift (dermatology)
- **DDI**: Alternative dermatology benchmark

Medical datasets have different characteristics — color/stain normalization is a long-standing problem, making style transfer TTA particularly relevant.

### 10.2 For Each New Dataset

The implementation agent must:
1. Check if data exists at `/data/local/retristyle/data/{dataset_name}/`
2. If not, download or provide instructions
3. Train classifiers (if not pretrained on relevant data)
4. Run all reference methods (DG baselines + geometric TTA)
5. Run style transfer TTA (using best configuration from ablations)
6. Compute all metrics and generate LaTeX tables

---

## 11. Infrastructure, Checkpointing & HPC Deployment

### 11.1 Docker / Apptainer Update

The existing Dockerfile is well-structured. Add:
- `open_clip` package to requirements.txt (for CLIP support)
- `umap-learn` package (for visualization)
- Ensure `faiss-gpu` or `faiss-cpu` is installed

### 11.2 $TMPDIR Staging for HPC (NHR@FAU)

**Critical performance optimization**: Stage datasets to node-local SSD (`$TMPDIR`) to avoid I/O bottleneck on `$WORK`.

Reference: https://doc.nhr.fau.de/data/filesystems/

Create a staging helper to be included in every SLURM script:

```bash
# ============================================================================
# Stage data to $TMPDIR (node-local SSD) for fast I/O
# ============================================================================
stage_dataset() {
    local DATASET_NAME=$1
    local SOURCE_DIR=$2      # e.g., $WORK/retristyle/data/$DATASET_NAME
    local STAGING_DIR="$TMPDIR/data/$DATASET_NAME"
    
    if [ -d "$STAGING_DIR" ]; then
        echo "Dataset already staged at $STAGING_DIR (shared from another job)"
        return 0
    fi
    
    echo "Staging $DATASET_NAME to $TMPDIR..."
    mkdir -p "$STAGING_DIR"
    
    # Check if source is a tar.gz archive (preferred for fast staging)
    if [ -f "${SOURCE_DIR}.tar.gz" ]; then
        echo "  Extracting archive ${SOURCE_DIR}.tar.gz → $STAGING_DIR"
        tar xzf "${SOURCE_DIR}.tar.gz" -C "$TMPDIR/data/"
    elif [ -f "${SOURCE_DIR}.tar" ]; then
        echo "  Extracting archive ${SOURCE_DIR}.tar → $STAGING_DIR"
        tar xf "${SOURCE_DIR}.tar" -C "$TMPDIR/data/"
    elif [ -d "$SOURCE_DIR" ]; then
        echo "  Copying directory $SOURCE_DIR → $STAGING_DIR"
        cp -r "$SOURCE_DIR" "$STAGING_DIR"
    else
        echo "  ERROR: No source found for $DATASET_NAME"
        return 1
    fi
    
    echo "  ✓ Staged: $(du -sh $STAGING_DIR | cut -f1)"
    return 0
}

# Stage the dataset
stage_dataset "$DATASET" "$WORK/retristyle/data/$DATASET"
DATA_PATH="$TMPDIR/data"   # Override data path to use staged data

# NOTE: For ImageNet-1k (~155GB), staging the full dataset may take >30 min
# and could exceed $TMPDIR capacity on some nodes.
# For ImageNet experiments, consider:
# 1. Only stage ImageNet-R/A/Sketch (small, <5GB each)
# 2. Keep ImageNet-1k on $WORK and only retrieve references on demand
#    (the ReferenceDatabase already loads lazily)
# 3. Pre-archive only the 200 relevant classes as imagenet1k_200classes.tar.gz
```

### 11.3 SLURM Script Generator Architecture

Instead of creating individual SLURM scripts, create **generator scripts** that produce them. Follow the existing pattern from `create_training_scripts_baseline.sh`.

Each generator:
1. Defines dataset/seed/method arrays
2. Creates one submission script per (dataset, seed, method) combination
3. Creates a top-level `submit_all.sh` that sbatch's all scripts
4. Includes $TMPDIR staging and auto-restart logic

**Key generators to create:**

```
scripts/thesis/
├── create_dataset_download.sh           # Download/verify datasets
├── create_domain_shift_analysis.sh      # Domain shift characterization
├── create_style_transfer_eval.sh        # Compare all 21 ST methods
├── create_classifier_training.sh        # Fine-tune classifiers (non-ImageNet)
├── create_reference_baselines.sh        # Train DG methods (DomainBed, SDG)
├── create_geometric_tta_eval.sh         # Geometric TTA baseline (ZERO, TPT, vanilla)
├── create_ablation_retrieval.sh         # Retrieval method ablation
├── create_ablation_aggregation.sh       # Aggregation strategy ablation
├── create_ablation_nrefs.sh             # N-refs sweep
├── create_generate_augmented_cache.sh   # Generate & cache stylized images
├── create_hybrid_tta_eval.sh            # Hybrid TTA mixing ratios
├── create_full_eval_extension.sh        # Extend best setup to all datasets
└── create_latex_tables_thesis.sh        # Generate LaTeX tables from results
```

### 11.4 SLURM Template Variables

Every generated SLURM script must include these configurable variables:

```bash
# ============================================================================
# CONFIGURABLE — edit for your cluster/setup
# ============================================================================
CONTAINER=$WORK/retristyle/retristyle-production.sif
DATA_PATH=$WORK/retristyle/data
OUTPUT_PATH=$WORK/retristyle/results
CHECKPOINT_PATH=$WORK/retristyle/checkpoints
EMBEDDING_DIR=$WORK/retristyle/embeddings
AUGMENTED_CACHE=$WORK/retristyle/augmented_cache
STYLE_WEIGHTS=/data/local/retristyle/models/style_transfer

# Proxy for internet access (NHR@FAU)
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
```

### 11.5 GPU Tier Selection

| Experiment Type | GPU Requirement | Partition |
|----------------|-----------------|-----------|
| Style transfer eval (small) | 1× A40/A100 | a40/a100 |
| Classifier training (ImageNet) | 4× A100 | a100 |
| Classifier training (small datasets) | 1× RTX3080/A40 | rtx3080/a40 |
| DomainBed training | 1-2× A40 | a40 |
| Geometric TTA inference | 1× RTX3080 | rtx3080 |
| RetriStyle TTA inference (n_refs ≤ 8) | 1× A40 | a40 |
| RetriStyle TTA inference (n_refs 16-64) | 2-4× A40 | a40 |
| Augmented image generation | 1-2× A40 | a40 |
| Embedding extraction (DINOv3) | 1× A40 | a40 |
| Domain shift analysis | 1× RTX3080 | rtx3080 |

### 11.6 Per-Sample Checkpointing

**Already fully implemented** in `experiments/tta/checkpoint.py`:

- Atomic writes: write to `.tmp`, rename to `.json`
- Per-sample persistence: predictions saved after every sample (configurable)
- Resume: on restart, counts existing predictions and skips processed samples
- Experiment key: unique per (classifier, method, strategy, retrieval, n_refs, augmentation, seed)

No changes needed for the basic pipeline. For the augmented image cache, apply the same pattern:

```python
# Track which test samples already have cached augmented images
processed = set(os.listdir(cache_dir / "sample_*"))
for sample_idx in range(total_samples):
    if f"sample_{sample_idx}" in processed:
        continue  # Already cached
    # Generate and save views
```

### 11.7 Wall-Time Management

For long-running jobs (especially ImageNet-R with 30,000 samples × 64 refs):

```bash
# Use 23h timeout with auto-resubmit (already in existing scripts)
timeout 23h apptainer exec ... python -m experiments.inference_tta ...

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 124 ]]; then
    echo "TIMEOUT: Resubmitting..."
    sbatch "${BASH_SOURCE[0]}"
fi
```

### 11.8 Optimized Dockerfile Additions

Update `requirements.txt` to add:

```
open-clip-torch>=2.24.0    # CLIP models
umap-learn>=0.5.0           # Visualization
seaborn>=0.13.0             # Plotting
```

---

## 12. Results Aggregation & LaTeX Reporting

### 12.1 Results Collection

Use the existing `experiments/tta/collect_results.py` to merge per-run JSON files:

```bash
python -m experiments.tta.collect_results \
    --results_dir $WORK/retristyle/results \
    --output_dir $WORK/retristyle/results/merged
```

### 12.2 Metrics Computation

Use existing `experiments/evaluate_predictions.py`:

```bash
python -m experiments.evaluate_predictions \
    --predictions_dir $WORK/retristyle/results/tta_inference/predictions/ \
    --output_path $WORK/retristyle/results/metrics/
```

### 12.3 LaTeX Table Generation

Create `scripts/thesis/create_latex_tables_thesis.sh` (generator) that produces scripts calling a new Python utility:

`experiments/thesis/generate_latex_tables.py`:

```python
# Reads merged results JSONs
# Generates LaTeX tables for:
#
# Table 1: Style Transfer Method Comparison
#   Rows: 21 methods  |  Cols: Wasserstein, SSIM, LPIPS, Edge, FID, ArtFID
#
# Table 2: Geometric TTA Baselines (ImageNet-R)
#   Rows: classifiers  |  Cols: No TTA, Vanilla, ZERO, TPT, TENT
#
# Table 3: Retrieval Method Ablation
#   Rows: retrieval strategies  |  Cols: Acc, BalAcc, AUC, ECE
#
# Table 4: Aggregation Strategy Ablation
#   Rows: vanilla/zero/tpt  |  Cols: Acc, BalAcc, AUC, ECE
#
# Table 5: N-refs Ablation
#   Rows: 2/4/8/16/32/64  |  Cols: Acc, BalAcc, AUC, ECE, Time
#
# Table 6: Hybrid TTA Mixing Ratios
#   Rows: mixing ratios  |  Cols: Acc, BalAcc, AUC, ECE
#
# Table 7: Per-Classifier Comparison (Best Style TTA vs Best Geometric TTA)
#   Rows: classifiers  |  Cols: No TTA, Geo-TTA, Style-TTA, Hybrid-TTA
#
# Table 8: Cross-Dataset Evaluation
#   Rows: datasets  |  Cols: No TTA, Geo-TTA, Style-TTA, Hybrid-TTA
#
# Table 9: DG Reference Methods + Our TTA
#   Rows: DG methods + ours  |  Cols: Per-dataset results
#
# Table 10: Per-Sample Complementarity Analysis
#   Counts of: both correct, geo-only correct, style-only correct, both wrong
```

### 12.4 Visualization Scripts

Create additional visualization utilities:

```python
# experiments/thesis/visualize_results.py
# - Bar plots comparing TTA methods across datasets
# - Per-class accuracy heatmaps
# - Confusion matrices
# - Sample qualitative examples (original → stylized → prediction)
# - t-SNE/UMAP domain gap visualizations
# - Complementarity Venn diagrams
```

---

## 13. Execution Order & Dependencies

### 13.1 Dependency Graph

```
Phase 0: Data Setup
  ├── Download ImageNet-1k (MANUAL — academic registration)
  ├── Download ImageNet-R, ImageNet-A, ImageNet-Sketch (automated)
  ├── Download PACS, VLCS (automated or manual)
  └── Check medical datasets

Phase 1: Domain Shift Analysis  [depends on: Phase 0]
  └── Run domain_shift_analysis.py on ImageNet-1k vs each variant
      Output: domain shift characterization + plots

Phase 2: Style Transfer Evaluation  [depends on: Phase 0]
  └── Run style_transfer_evaluation.py with all 21 methods
      Output: best style transfer method (ideally StyleID)

Phase 3: Geometric TTA Baseline  [depends on: Phase 0]
  ├── Run geometric TTA with pretrained classifiers on ImageNet-R
  ├── Run TENT baseline
  └── Run no-TTA baseline (single forward pass)
      Output: baseline accuracy per classifier, per-sample predictions

Phase 4: Ablation Suite  [depends on: Phase 2, Phase 3]
  ├── Ablation 1: Retrieval method (6 strategies)
  │   Output: best retrieval strategy
  ├── Ablation 2: Aggregation strategy (3 strategies)
  │   Output: best aggregation strategy
  ├── Ablation 3: N-refs sweep (6 values)
  │   Output: best n_refs value
  └── Generate augmented image cache
      Output: cached views on disk

Phase 5: Hybrid TTA  [depends on: Phase 3, Phase 4]
  └── Evaluate 5 mixing ratios
      Output: best hybrid ratio, complementarity analysis

Phase 6: Multi-Classifier Evaluation  [depends on: Phase 4 cached images]
  └── Run all classifiers on cached augmented views
      Output: per-classifier results

Phase 7: Extension to ImageNet Variants  [depends on: Phase 4 best setup]
  ├── ImageNet-A
  ├── ImageNet-C (15 corruptions × 5 severities)
  ├── ImageNet-Sketch
  └── ImageNet-V2

Phase 8: PACS & VLCS  [depends on: Phase 7 methodology]
  ├── Train classifiers per domain  (requires cluster)
  ├── Train DG baselines per domain  (requires cluster)
  └── Run TTA evaluation per domain

Phase 9: Medical Datasets  [depends on: Phase 7 methodology]
  ├── Train classifiers per split
  ├── Train DG baselines per split
  └── Run TTA evaluation per split

Phase 10: Results & Reporting  [depends on: all previous]
  ├── Collect & merge all results
  ├── Generate LaTeX tables
  ├── Generate visualizations
  └── Produce complementarity analysis
```

### 13.2 Estimated Computational Budget

| Phase | # Jobs | GPU Hours (est.) | Priority |
|-------|--------|-----------------|----------|
| Domain shift analysis | 5-10 | ~10h | P1 |
| Style transfer eval | 21 methods × 800 pairs | ~40h | P1 |
| Geometric TTA baseline | 6 classifiers × 3 strategies × 3 seeds | ~50h | P1 |
| Ablation 1 (retrieval) | 6 strategies × 3 seeds | ~100h | P1 |
| Ablation 2 (aggregation) | 3 strategies × 3 seeds | ~50h | P1 |
| Ablation 3 (n_refs) | 6 values × 3 seeds | ~100h | P1 |
| Augmented cache generation | 30K samples × 64 views | ~200h | P1 |
| Hybrid TTA | 5 ratios × 3 seeds | ~50h | P1 |
| Multi-classifier | 5 classifiers × 3 seeds | ~50h | P1 |
| ImageNet variants | 4 variants × 6 classifiers × 3 seeds | ~200h | P2 |
| PACS (training) | 4 domains × 6 classifiers × 6+1 methods × 3 seeds | ~500h | P2 |
| VLCS (training) | 4 domains × 6 classifiers × 6+1 methods × 3 seeds | ~500h | P3 |
| Medical (training) | 2 datasets × 6 classifiers × 3 seeds | ~100h | P3 |

### 13.3 Implementation Agent Checklist

The implementation agent should work through these tasks in order:

- [ ] **0.1** Create `scripts/thesis/` directory structure
- [ ] **0.2** Create dataset download/verification script — check `/data/local/retristyle/data/` for each dataset
- [ ] **0.3** Download ImageNet-R, ImageNet-A (automated wget scripts)
- [ ] **0.4** Provide manual download instructions for ImageNet-1k
- [ ] **0.5** Update `requirements.txt` with open-clip-torch, umap-learn
- [ ] **0.6** Update Dockerfile if needed
- [ ] **0.7** Add thesis constants to `experiments/tta/constants.py`
- [ ] **1.1** Implement `experiments/thesis/domain_shift_analysis.py` (FID, Wasserstein, Gram, edge, MMD, t-SNE)
- [ ] **1.2** Create corresponding script generator `scripts/thesis/create_domain_shift_analysis.sh`
- [ ] **2.1** Adapt `experiments/reference_methods/color_transfer.py` → `experiments/thesis/style_transfer_evaluation.py` for the 20×40 protocol
- [ ] **2.2** Create script generator `scripts/thesis/create_style_transfer_eval.sh`
- [ ] **3.1** Implement CLIP classifier wrapper (zero-shot or linear probe via open_clip)
- [ ] **3.2** Implement DINOv3 linear-probe classifier head
- [ ] **3.3** Create script generator for classifier fine-tuning on non-ImageNet datasets
- [ ] **4.1** Create script generator `scripts/thesis/create_geometric_tta_eval.sh` for all baselines
- [ ] **4.2** Create script generators for all three ablation studies (retrieval, aggregation, n_refs)
- [ ] **5.1** Implement `experiments/thesis/generate_augmented_images.py` with disk caching + checkpointing
- [ ] **5.2** Add `--augmented_cache` parameter to `run_inference.py` to load cached views
- [ ] **6.1** Implement `experiments/thesis/hybrid_tta.py` (or extend `run_inference.py` with `--hybrid_geo_ratio`)
- [ ] **6.2** Create script generator for hybrid TTA evaluation
- [ ] **7.1** Create script generators for extension to other ImageNet test sets
- [ ] **7.2** Create script generators for PACS/VLCS full protocol
- [ ] **7.3** Create script generators for medical datasets
- [ ] **8.1** Implement `experiments/thesis/generate_latex_tables.py`
- [ ] **8.2** Implement `experiments/thesis/visualize_results.py`
- [ ] **8.3** Create script generator `scripts/thesis/create_latex_tables_thesis.sh`
- [ ] **9.1** Add $TMPDIR staging helper to all SLURM script generators
- [ ] **9.2** Ensure all scripts have auto-restart on timeout
- [ ] **9.3** Update `prepare-hpc.sh` for thesis configuration
- [ ] **10.1** Per-sample complementarity analysis script (geometric vs style TTA)
- [ ] **10.2** Class-level analysis of TTA effectiveness

### 13.4 Placeholder Convention for Bash Scripts

For experiments that depend on earlier results (e.g., "best retrieval strategy"), use this placeholder pattern:

```bash
# ============================================================================
# PLACEHOLDER — Update after Ablation 1 results
# ============================================================================
# Current: ZERO default setup
# After ablation, replace with best configuration found
BEST_RETRIEVAL=${BEST_RETRIEVAL:-"dino"}           # Placeholder: update after Abl. 1
BEST_AGGREGATION=${BEST_AGGREGATION:-"zero"}        # Placeholder: update after Abl. 2
BEST_N_REFS=${BEST_N_REFS:-64}                      # Placeholder: update after Abl. 3
BEST_HYBRID_RATIO=${BEST_HYBRID_RATIO:-"0.5"}       # Placeholder: update after Abl. 5
```

This allows scripts to be generated and submitted with reasonable defaults, then re-generated with the actual best values once results are in. The environment variable override (`${VAR:-default}`) makes it easy to switch without editing the scripts.

---

## Appendix A: Complete Style Transfer Methods Inventory

| Method | Category | Weight File | Native Size | Notes |
|--------|----------|-------------|-------------|-------|
| AdaIN | Artistic (trained) | adain.pth | 256 | Fast, simple baseline |
| AdaAttN | Artistic (trained) | adaattn.pth | 512 | Attention-based |
| AesPANet | Artistic (trained) | aespanet.pth | 256 | Aesthetic-aware |
| ArtFlow | Artistic (trained) | artflow.pth | 256 | Flow-based |
| CAST | Artistic (trained) | cast.pth | 256 | Contrastive |
| EFDM | Artistic (trained) | efdm.pth | 256 | Feature distribution |
| IEContrAST | Artistic (trained) | iecontrast.pth | 256 | Contrastive |
| MAST | Artistic (trained) | mast.pth | 512 | Manifold-aligned |
| SANet | Artistic (trained) | sanet.pth | 256 | Style attentional |
| StyleFormer | Artistic (trained) | styleformer.pth | 256 | Transformer-based |
| StyTr2 | Artistic (trained) | stytr2.pth | 256 | Transformer-based |
| StyleID | Diffusion (free) | — (SD 1.4 auto-dl) | 512 | **Flagship** CVPR 2024 |
| StyleSSP | Diffusion (free) | — (SD auto-dl) | 512 | SDXL-based |
| InstantStyle | Diffusion (free) | — (auto-dl) | 512 | ControlNet-based |
| DiffStyle | Diffusion (free) | diffstyle.pt | 256 | Classifier guidance |
| DiffuseIT | Diffusion (free) | diffuseit.pt | 256 | CLIP-guided |
| ModFlows | Photorealistic | modflows.pt | 256 | Normalizing flows |
| WCT2 | Photorealistic | wct2.pt | 256 | Wavelet |
| DeepPreset | Photorealistic | deeppreset.tar | 256 | Preset-based |
| PhotoNAS | Photorealistic | photonas.pth.tar | 256 | NAS-optimized |

## Appendix B: Key Existing Functions to Reuse

| Function / Class | Location | Purpose |
|-----------------|----------|---------|
| `create_dataset()` | `experiments/data/_factory.py` | Universal dataset loader |
| `load_classifier()` | `experiments/classifier_evaluation.py` | Load any timm model |
| `compute_metrics()` | `experiments/classifier_evaluation.py` | Acc, BalAcc, AUC |
| `compute_ece()` | `experiments/classifier_evaluation.py` | ECE calibration |
| `create_color_transfer_method()` | `experiments/reference_methods/color_transfer_factory.py` | Load any ST method |
| `StyleIDMethod` | `retristyle/infer_style_base.py` | StyleID style transfer |
| `ReferenceDatabase` | `retristyle/retrieval/reference_db.py` | Lazy reference loading |
| `DinoRetriever` | `retristyle/retrieval/dino_retriever.py` | DINO+FAISS retrieval |
| `build_reference_db()` | `experiments/tta/reference_db_setup.py` | Build retrieval DB |
| `build_retriever()` | `experiments/tta/reference_db_setup.py` | Retriever factory |
| `augment_views()` | `experiments/tta/augmentation.py` | Generate TTA views |
| `eval_zero()` / `eval_tpt()` | `experiments/tta/evaluation.py` | Aggregation strategies |
| `load_predictions()` / `save_predictions()` | `experiments/tta/checkpoint.py` | Checkpointed I/O |
| `extract_and_cache()` | `experiments/tta/extract_embeddings.py` | DINO embedding extraction |
| `collect_results()` | `experiments/tta/collect_results.py` | Merge per-run JSONs |

## Appendix C: $TMPDIR Staging Template

For inclusion in every HPC SLURM script:

```bash
# ============================================================================
# $TMPDIR Data Staging for Fast I/O
# Reference: https://doc.nhr.fau.de/data/filesystems/
# ============================================================================
# $TMPDIR is a node-local NVMe SSD (Alex, TinyGPU, Woody) or RAM disk (Fritz)
# Auto-created per job, auto-cleaned at job end
# Shared between jobs on the same node if running simultaneously

STAGING_ENABLED=${STAGING_ENABLED:-true}

if [[ "$STAGING_ENABLED" == "true" ]] && [[ -n "$TMPDIR" ]]; then
    echo "Staging data to \$TMPDIR ($TMPDIR)..."
    
    # Create staging directory
    mkdir -p "$TMPDIR/data"
    
    # Stage dataset (prefer .tar.gz archives for speed)
    if [[ -f "$WORK/retristyle/data/${DATASET}.tar.gz" ]]; then
        echo "  Extracting ${DATASET}.tar.gz to $TMPDIR/data/"
        tar xzf "$WORK/retristyle/data/${DATASET}.tar.gz" -C "$TMPDIR/data/" &
        STAGE_PID=$!
    elif [[ -f "$WORK/retristyle/data/${DATASET}.tar" ]]; then
        echo "  Extracting ${DATASET}.tar to $TMPDIR/data/"
        tar xf "$WORK/retristyle/data/${DATASET}.tar" -C "$TMPDIR/data/" &
        STAGE_PID=$!
    elif [[ -d "$WORK/retristyle/data/${DATASET}" ]]; then
        echo "  Copying ${DATASET} to $TMPDIR/data/"
        cp -r "$WORK/retristyle/data/${DATASET}" "$TMPDIR/data/" &
        STAGE_PID=$!
    fi
    
    # Wait for staging to complete
    if [[ -n "${STAGE_PID:-}" ]]; then
        wait $STAGE_PID
        echo "  ✓ Staging complete: $(du -sh $TMPDIR/data/${DATASET} 2>/dev/null | cut -f1)"
    fi
    
    # Use staged data for I/O-heavy operations
    STAGED_DATA_PATH="$TMPDIR/data"
else
    echo "Staging disabled or \$TMPDIR not set — using \$WORK directly"
    STAGED_DATA_PATH="$WORK/retristyle/data"
fi
```

---

*END OF IMPLEMENTATION PLAN*
