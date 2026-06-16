# ============================================================================
# Seeds
# ============================================================================

DEFAULT_SEED = 71397589

ALL_SEEDS = [
    71397589,
    133560673,
    265017005,
]

# ============================================================================
# Classifiers
# ============================================================================

CNN_CLASSIFIERS = [
    "resnet18",
    "densenet121",
]

VIT_CLASSIFIERS = [
    "vit_base_patch16_224",
    "swin_base_patch4_window7_224",
]

VLM_CLASSIFIERS = [
    "ViT-B-16",
    "ViT-B-16@Zero",
]

FM_CLASSIFIERS = [
    "dinov2_vitb14",
]

ALL_CLASSIFIERS = (
    CNN_CLASSIFIERS
    + VIT_CLASSIFIERS
    + VLM_CLASSIFIERS
    + FM_CLASSIFIERS
)

PRETRAINED_CLASSIFIERS = {
    "ViT-B-16",
    "dinov2_vitb14",
    "vit_base_patch16_dinov3_lvd1689m",
}

# ============================================================================
# Retrieval
# ============================================================================

EMBEDDING_MODEL = "vit_base_patch16_dinov3.lvd1689m"

RETRIEVAL_STRATEGIES = [
    "random",
    "balanced_random",
    "dino",
]

EVAL_STRATEGIES = [
    "vanilla",
    "zero",
    "tpt",
]

N_REFS_VALUES = [2, 4, 8, 16, 32]

DEFAULT_N_VIEWS = 16

STYLE_BATCH_SIZE = 8

# ============================================================================
# Hybrid TTA
# ============================================================================

TTA_STRATEGIES = [
    "retristyle",
    "geometric_tta",
    "adain_tta"
]
GEO_FRACS = [
    1.0,
    0.75,
    0.5,
    0.25,
    0.0,
]

# ============================================================================
# Augmentations
# ============================================================================

TRAINING_AUGMENTATIONS = [
    "none",
    "color_jitter",
    "rand_augment",
    "trivial_augment",
    "aug_mix",
]
