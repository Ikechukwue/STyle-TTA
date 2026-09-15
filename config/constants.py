
import matplotlib.pyplot as plt
# ============================================================================
# Seeds
# ============================================================================

DEFAULT_SEED = 71397589

ALL_SEEDS = [
    71397589,
    133560673,
    265017005,
]

ALL_METHODS = {
    "ablation/retristyle": "STyle-TTA(StyleID)",
    "ablation/adain": "STyle-TTA(AdaIN)",
    "geometric_tta": "Geometric TTA",
    "hybrid_tta": "Hybrid-TTA(Geo-Sty)"
}
ALL_DATASETS = {
    "midog": "MIDOG",
    "eurosat": "EuroSAT",
    "camelyon17wilds": "Camelyon17-WILDS",
    "epistr": "EpiStr",
    "imagenet": "ImageNet-1K",
}

ALL_SPLITS = {
    "midog": "MIDOG(Test)",
    "eurosat": "UCMerced",
    "camelyon17wilds": "Camelyon17-WILDS(Test)",
    "epistr": "EpiStr(Test)",
    "imagenet": "ImageNet-R",
}

TRUE_SPLITS = {
    "MIDOG(Test)": "test",
    "UCMerced": "ucmerced",
    "Camelyon17-WILDS(Test)": "test",
    "EpiStr(Test)": "test",
    "ImageNet-R": "test_r",
}

CLEAN_CLASSIFIERS = {
        "resnet18": ("ResNet-18", "CNN"),
        "densenet121": ("DenseNet-121", "CNN"),
        "vit_base_patch16_224": ("ViT-B/16 (224)", "Vision Transformer"),
        "swin_base_patch4_window7_224": ("Swin-B (224)", "Vision Transformer"),
        "ViT-B-16": ("CLIP ViT-B/16", "Foundation Model"),
        "ViT-B-16@Zero": ("CLIP ViT-B/16 (Zero-Shot)", "Vision-Language Model"),
        "dinov2_vitb14": ("DINOv2 ViT-B/14", "Foundation Model"),
        "vit_base_patch16_dinov3_lvd1689m": ("DINOv3 ViT-B/16", "Foundation Model"),
    }
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

DOMAIN_METRICS={"mmd": "Mean Centroid Distance", 
                "wasserstein": "Wasserstein",
                "kl_symmetric": "Symmetric-KL"}
                
N_REFS_VALUES = [1, 2, 4, 8, 16, 32]

DEFAULT_N_VIEWS = 16

STYLE_BATCH_SIZE = 8

# ============================================================================
# Hybrid TTA
# ============================================================================

TTA_STRATEGIES = {
        "ablation/adain_tta": {"template": "{cl}_adain_tta_{eval}_{retr}_nrefs{rfs}_seed{seed}.json", "default_eval": "zero", "default_retr": "dino","axis": [2, 4, 8, 16, 32, 64], "color": "tab:red","color_shade": "Reds", "label": "AdaIN"},
        "ablation/retristyle": {"template": "{cl}_retristyle_{eval}_dino_nrefs{rfs}_seed{seed}.json", "default_eval":"vanilla", "default_retr": "dino", "axis": [2, 4, 8, 16], "color": "tab:green","color_shade": "Greens", "label": "STyle-TTA"},
        "geometric_tta": {"template": "{cl}_geometric_{eval}_nviews{rfs}_seed{seed}.json{retr}", "axis": [2, 4, 8, 16, 32, 64], "color": "tab:blue", "color_shade": "Blues","label": "Geometric"},
        "hybrid_tta": {"template": "{ds}_{cl}_hybrid_geo{geo}_sty{sty}_{eval}_split{use_n}_nr{rfs}_seed{seed}_results.json","default_eval":"vanilla", "default_retr":"dino", "axis": [4, 8, 16, 32, 64], "color": "tab:purple", "color_shade": "Purples","label": "Hybrid(Style/Geo)"},
    }
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

