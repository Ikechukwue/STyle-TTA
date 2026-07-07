"""
Shared constants and enumerations for TTA inference.
"""

from __future__ import annotations

# Reproducibility
DEFAULT_SEED = 71397589
ALL_SEEDS = [71397589, 133560673, 265017005]

# ZERO evaluation strategy defaults (Farina et al., NeurIPS 2024)
ZERO_N_VIEWS = 64
ZERO_GAMMA = 0.3  # retain top 30 % most confident views

# TPT evaluation strategy defaults (Shu et al., NeurIPS 2022)
# Confidence filter retains views whose entropy is in the bottom-10 %
# percentile, then averages their softmax predictions.
TPT_GAMMA = 0.1  # retain top 10 % most confident views

# Default number of stochastic views for augmentation-based TTA methods
DEFAULT_N_VIEWS = 64

# ── Augmentation-based TTA methods ──────────────────────────────────────────
# Each generates N stochastic views by repeatedly applying the augmentation.
AUGMENTATION_TTA_METHODS = [
    "geometric",           # ZERO paper: random resized crop + horizontal flip
    "gray_scale",
    "color_jitter",
    "auto_augment",
    "rand_augment",
    "trivial_augment",
    "aug_mix",
    "random_resized_crop",
    "random_flip",
    "random_erasing",
    "targeted_augment",
    "oracle",              # randomly picks one of the above per view
]

# ── Style-transfer / retrieval-based TTA methods ───────────────────────────
RETRIEVAL_TTA_METHODS = [
    "adain_tta",
    "color_tta",
    "retristyle",
]

# Available options -------------------------------------------------------
AVAILABLE_TTA_METHODS = [
    *AUGMENTATION_TTA_METHODS,
    "tent",
    *RETRIEVAL_TTA_METHODS,
    "hybrid_tta",
]

AVAILABLE_EVAL_STRATEGIES = ["vanilla", "zero", "tpt", "foods"]

AVAILABLE_RETRIEVAL_STRATEGIES = [
    "random",
    "balanced_random",
    "metric",
    "balanced_metric",
    "dino",
]

# Training augmentation names (for constructing weights paths)
TRAINING_AUGMENTATIONS = [
    "none",
    "gray_scale",
    "color_jitter",
    "auto_augment",
    "rand_augment",
    "trivial_augment",
    "aug_mix",
    "random_resized_crop",
    "random_flip",
    "random_erasing",
    "targeted_augment",
]

# ── Thesis-specific constants ───────────────────────────────────────────────
# Primary ablation setup: ImageNet-1k → ImageNet-R
THESIS_PRIMARY_DATASET = "imagenet"
THESIS_PRIMARY_TRAIN_SPLIT = "train"
THESIS_PRIMARY_TEST_SPLIT = "test_r"

THESIS_CLASSIFIERS = {
    "cnn": ["resnet18", "densenet121"],
    "vit": ["vit_base_patch16_224", "swin_base_patch4_window7_224"],
    "vlm": ["ViT-B-16"],  # CLIP (via open_clip)
    "fm": ["dinov2_vitb14"],  # DINOv2
}

# Flat list for iteration
THESIS_ALL_CLASSIFIERS = [
    "resnet18", "densenet121",
    "vit_base_patch16_224", "swin_base_patch4_window7_224",
    "ViT-B-16",  # CLIP
    "dinov2_vitb14",  # DINOv2
]
CACHE_VIEW_CLASSIFIERS = ["ViT-B-16", "dinov2_vitb14"]
THESIS_N_REFS_SWEEP = [2, 4, 8, 16, 32, 64]

THESIS_HYBRID_RATIOS = [
    (1.0, 0.0),    # pure geometric
    (0.75, 0.25),  # 3/4 geo, 1/4 style
    (0.5, 0.5),    # half and half
    (0.25, 0.75),  # 1/4 geo, 3/4 style
    (0.0, 1.0),    # pure style
]

THESIS_SEEDS = [71397589, 133560673, 265017005]

# ImageNet-R has 200 classes out of 1000
IMAGENET_R_NUM_CLASSES = 200
