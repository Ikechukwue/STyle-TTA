"""
xAILab Bamberg
University of Bamberg

@description:
Constant values.
"""

# ============================================
# Mean and std
# ============================================
# Mean calculated from the training sets
NORMALIZATION_MEAN = {
    # Camelyon17WILDS (default dataset splits)
    "camelyon17wilds": (0.7440, 0.5895, 0.7214),

    # Epithelium-Stroma (dataset splits across datasets)
    "epistr": (0.7360, 0.5158, 0.8072),  # Train: NKI / Val: VGH / Test: IHC

    # Peripheral Blood (combines MLL23, Acevedo20 (for train/val), and Matek19 (test))
    "peripheral_blood": (0.7874, 0.6866, 0.7588),

    # bone marrow smears and peripheral blood smears
    "bone_marrow_smears_and_peripheral_blood": (0.5531, 0.4894, 0.7275),

    # Fitzpatrick17k (dataset splits across skin tones)
    "fitzpatrick17k": (0.6219, 0.4917, 0.4478),  # Train: 1,2 / Val: 3,4 / Test: 5,6

    # DDI (dataset splits across skin tones)
    "ddi": (0.1831, 0.1881, 0.1985),  # Train: 12 / Val: 34 / Test: 56

    # Retina (OOD: Train: APTOS+DeepDR / Val: IDRiD / Test: MESSIDOR-2)
    "retina": (0.3953, 0.2224, 0.0987),

    # MedMNIST 2D datasets
    "pathmnist": (0.7405, 0.5329, 0.7058),
    "chestmnist": (0.4979, 0.4979, 0.4979),
    "dermamnist": (0.7632, 0.5381, 0.5615),
    "octmnist": (0.1896, 0.1896, 0.1896),
    "pneumoniamnist": (0.5717, 0.5717, 0.5717),
    "retinamnist": (0.3945, 0.2416, 0.1453),
    "breastmnist": (0.3277, 0.3277, 0.3277),
    "bloodmnist": (0.7961, 0.6596, 0.6964),
    "tissuemnist": (0.1021, 0.1021, 0.1021),
    "organamnist": (0.4680, 0.4680, 0.4680),
    "organcmnist": (0.4942, 0.4942, 0.4942),
    "organsmnist": (0.4952, 0.4952, 0.4952),

    # --- Domain Generalization Datasets ---

    # CIFAR-10 (computed from training set)
    "cifar10": (0.4914, 0.4822, 0.4465),
    # CIFAR-100 (computed from training set)
    "cifar100": (0.5071, 0.4867, 0.4408),

    # ImageNet-1k (standard values)
    "imagenet": (0.485, 0.456, 0.406),

    # PACS (using ImageNet stats as proxy for natural images)
    "pacs": (0.485, 0.456, 0.406),
    # VLCS
    "vlcs": (0.485, 0.456, 0.406),
    # DomainNet
    "domainnet": (0.485, 0.456, 0.406),
    # Office-Home
    "office_home": (0.485, 0.456, 0.406),
    # Office-Caltech
    "office_caltech": (0.485, 0.456, 0.406),
    # VisDA-17
    "visda17": (0.485, 0.456, 0.406),
    # Visual Decathlon
    "visual_decathlon": (0.485, 0.456, 0.406),
    # Terra Incognita
    "terra_incognita": (0.485, 0.456, 0.406),

    # Colored MNIST (grayscale-based, computed from MNIST)
    "colored_mnist": (0.1307, 0.1307, 0.1307),

    # NICO / NICO++ (using ImageNet stats as proxy)
    "nico": (0.485, 0.456, 0.406),
    # MetaShift (using ImageNet stats as proxy)
    "metashift": (0.485, 0.456, 0.406),
    # OpenMIBOOD (placeholder - recompute per benchmark)
    "openmibood": (0.485, 0.456, 0.406),
}

# STD calculated from the training sets
NORMALIZATION_STD = {
    # Camelyon17WILDS (default dataset splits)
    "camelyon17wilds": (0.1787, 0.2131, 0.1721),

    # Epithelium-Stroma (dataset splits across datasets)
    "epistr": (0.1948, 0.2434, 0.1438),  # Train: NKI / Val: VGH / Test: IHC

    # Peripheral Blood (combines MLL23, Acevedo20 (for train/val), and Matek19 (test))
    "peripheral_blood": (0.2265, 0.2598, 0.0962),

    # bone marrow smears and peripheral blood smears
    "bone_marrow_smears_and_peripheral_blood": (0.2427, 0.283, 0.1777),

    # Fitzpatrick17k (dataset splits across skin tones)
    "fitzpatrick17k": (0.2279, 0.1982, 0.1991),  # Train: 1,2 / Val: 3,4 / Test: 5,6

    # DDI (dataset splits across skin tones)
    "ddi": (0.1831, 0.1881, 0.1985),  # Train: 12 / Val: 34 / Test: 56

    # Retina (OOD: Train: APTOS+DeepDR / Val: IDRiD / Test: MESSIDOR-2)
    "retina": (0.2806, 0.1649, 0.1091),

    # MedMNIST 2D datasets
    "pathmnist": (0.1651, 0.2174, 0.1574),
    "chestmnist": (0.2480, 0.2480, 0.2480),
    "dermamnist": (0.1368, 0.1587, 0.1769),
    "octmnist": (0.2148, 0.2148, 0.2148),
    "pneumoniamnist": (0.1770, 0.1770, 0.1770),
    "retinamnist": (0.3238, 0.2106, 0.1513),
    "breastmnist": (0.2194, 0.2194, 0.2194),
    "bloodmnist": (0.2265, 0.2598, 0.0962),
    "tissuemnist": (0.0989, 0.0989, 0.0989),
    "organamnist": (0.2876, 0.2876, 0.2876),
    "organcmnist": (0.2776, 0.2776, 0.2776),
    "organsmnist": (0.2769, 0.2769, 0.2769),

    # --- Domain Generalization Datasets ---

    # CIFAR-10 (computed from training set)
    "cifar10": (0.2470, 0.2435, 0.2616),
    # CIFAR-100 (computed from training set)
    "cifar100": (0.2675, 0.2565, 0.2761),

    # ImageNet-1k (standard values)
    "imagenet": (0.229, 0.224, 0.225),

    # PACS (using ImageNet stats as proxy for natural images)
    "pacs": (0.229, 0.224, 0.225),
    # VLCS
    "vlcs": (0.229, 0.224, 0.225),
    # DomainNet
    "domainnet": (0.229, 0.224, 0.225),
    # Office-Home
    "office_home": (0.229, 0.224, 0.225),
    # Office-Caltech
    "office_caltech": (0.229, 0.224, 0.225),
    # VisDA-17
    "visda17": (0.229, 0.224, 0.225),
    # Visual Decathlon
    "visual_decathlon": (0.229, 0.224, 0.225),
    # Terra Incognita
    "terra_incognita": (0.229, 0.224, 0.225),

    # Colored MNIST (grayscale-based, computed from MNIST)
    "colored_mnist": (0.3081, 0.3081, 0.3081),

    # NICO / NICO++ (using ImageNet stats as proxy)
    "nico": (0.229, 0.224, 0.225),
    # MetaShift (using ImageNet stats as proxy)
    "metashift": (0.229, 0.224, 0.225),
    # OpenMIBOOD (placeholder - recompute per benchmark)
    "openmibood": (0.229, 0.224, 0.225),
}

# ============================================
# Number of classes
# ============================================
NUM_CLASSES = {
    # Camelyon17WILDS
    "camelyon17wilds": 2,

    # Epithelium-Stroma
    "epistr": 2,

    # Peripheral Blood (combines MLL23, Acevedo20 (for train/val), and Matek19 (test))
    "peripheral_blood": 13,

    # bone marrow smears and peripheral blood smears
    "bone_marrow_smears_and_peripheral_blood": 13,

    # Fitzpatrick17k (Possible classification tasks: (3, 9, 114))
    "fitzpatrick17k": 3,

    # DDI
    "ddi": 2,

    # Retina (Diabetic Retinopathy grading: 0-4)
    "retina": 5,

    # MedMNIST 2D datasets (from MedMNIST INFO)
    "pathmnist": 9,        # Colon Pathology - 9 tissue types
    "chestmnist": 14,      # Chest X-Ray - 14 diseases (multi-label)
    "dermamnist": 7,       # Dermatoscopy - 7 skin diseases
    "octmnist": 4,         # Retinal OCT - 4 conditions
    "pneumoniamnist": 2,   # Chest X-Ray Pneumonia - binary
    "retinamnist": 5,      # Fundus Photography - 5 grades
    "breastmnist": 2,      # Breast Ultrasound - binary
    "bloodmnist": 8,       # Blood Cell Microscopy - 8 cell types
    "tissuemnist": 8,      # Kidney Cortex Microscopy - 8 tissue types
    "organamnist": 11,     # Abdominal CT - 11 organs
    "organcmnist": 11,     # Abdominal CT - 11 organs
    "organsmnist": 11,     # Abdominal CT - 11 organs

    # --- Domain Generalization Datasets ---
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 1000,
    "pacs": 7,
    "vlcs": 5,
    "domainnet": 345,
    "office_home": 65,
    "office_caltech": 10,
    "visda17": 12,
    "visual_decathlon": 10,  # Varies per task; default imagenet12 has 1000
    "terra_incognita": 10,
    "colored_mnist": 2,
    "nico": 10,              # Varies; set per NICO subset used
    "metashift": 2,          # Varies; set per MetaShift subset generated
    "openmibood": 2,         # Varies per benchmark; set accordingly
}

# ============================================
# Task types
# ============================================
# Task type for each dataset: "multi-class" or "multi-label"
TASK_TYPE = {
    # Non-MedMNIST datasets (all multi-class)
    "camelyon17wilds": "multi-class",
    "epistr": "binary-class",
    "peripheral_blood": "multi-class",
    "bone_marrow_smears_and_peripheral_blood": "multi-class",
    "fitzpatrick17k": "multi-class",
    "ddi": "binary-class",
    "retina": "ordinal",

    # MedMNIST 2D datasets
    "pathmnist": "multi-class",
    "chestmnist": "multi-label", 
    "dermamnist": "multi-class",
    "octmnist": "multi-class",
    "pneumoniamnist": "binary-class",
    "retinamnist": "ordinal",
    "breastmnist": "binary-class",
    "bloodmnist": "multi-class",
    "tissuemnist": "multi-class",
    "organamnist": "multi-class",
    "organcmnist": "multi-class",
    "organsmnist": "multi-class",

    # --- Domain Generalization Datasets ---
    "cifar10": "multi-class",
    "cifar100": "multi-class",
    "imagenet": "multi-class",
    "pacs": "multi-class",
    "vlcs": "multi-class",
    "domainnet": "multi-class",
    "office_home": "multi-class",
    "office_caltech": "multi-class",
    "visda17": "multi-class",
    "visual_decathlon": "multi-class",
    "terra_incognita": "multi-class",
    "colored_mnist": "binary-class",
    "nico": "multi-class",
    "metashift": "multi-class",
    "openmibood": "multi-class",
}

# ============================================
# Available dataset splits
# ============================================
DATASET_SPLITS = {
    # Camelyon17WILDS
    "camelyon17wilds": ["train", "val", "id_val", "test"],

    # Epithelium-Stroma
    "epistr": ["train", "val", "test"],

    # Peripheral Blood
    "peripheral_blood": ["train", "val", "test"],

    # bone marrow smears and peripheral blood smears
    "bone_marrow_smears_and_peripheral_blood": ["train", "val", "test"],

    # Fitzpatrick17k
    "fitzpatrick17k": ["train", "val", "test"],

    # DDI
    "ddi": ["train", "val", "test"],

    # Retina
    "retina": ["train", "val", "test"],

    # MedMNIST 2D datasets (standard train/val/test splits)
    "pathmnist": ["train", "val", "test"],
    "chestmnist": ["train", "val", "test"],
    "dermamnist": ["train", "val", "test"],
    "octmnist": ["train", "val", "test"],
    "pneumoniamnist": ["train", "val", "test"],
    "retinamnist": ["train", "val", "test"],
    "breastmnist": ["train", "val", "test"],
    "bloodmnist": ["train", "val", "test"],
    "tissuemnist": ["train", "val", "test"],
    "organamnist": ["train", "val", "test"],
    "organcmnist": ["train", "val", "test"],
    "organsmnist": ["train", "val", "test"],

    # --- Domain Generalization Datasets ---
    # CIFAR-10 / CIFAR-100 (clean train/val + clean test + corrupted test)
    "cifar10": ["train", "val", "test", "test_c"],
    "cifar100": ["train", "val", "test", "test_c"],

    # ImageNet-1k + variants
    "imagenet": ["train", "val", "test", "test_a", "test_r", "test_c",
                 "test_p", "test_sketch", "test_v2", "test_abl"],

    # PACS (default train_domain=photo)
    "pacs": ["train", "val", "test",
             "test_art_painting", "test_cartoon", "test_photo", "test_sketch"],

    # VLCS (default train_domain=PASCAL)
    "vlcs": ["train", "val", "test",
             "test_CALTECH", "test_LABELME", "test_PASCAL", "test_SUN"],

    # DomainNet (default train_domain=real)
    "domainnet": ["train", "val", "test",
                  "test_clipart", "test_infograph", "test_painting",
                  "test_quickdraw", "test_real", "test_sketch"],

    # Office-Home (default train_domain=Product)
    "office_home": ["train", "val", "test",
                    "test_Art", "test_Clipart", "test_Product", "test_Real_World"],

    # Office-Caltech (default train_domain=amazon)
    "office_caltech": ["train", "val", "test",
                       "test_amazon", "test_caltech", "test_dslr", "test_webcam"],

    # VisDA-17 (default train_domain=train)
    "visda17": ["train", "val", "test",
                "test_train", "test_validation", "test_test"],

    # Visual Decathlon (default train_domain=imagenet12)
    "visual_decathlon": ["train", "val", "test"],

    # Terra Incognita (default train_domain=location_38)
    "terra_incognita": ["train", "val", "test",
                        "test_location_38", "test_location_43",
                        "test_location_46", "test_location_100"],

    # Colored MNIST (default train_domain=env0)
    "colored_mnist": ["train", "val", "test",
                      "test_env0", "test_env1", "test_env2"],

    # NICO / NICO++ (contexts auto-discovered)
    "nico": ["train", "val", "test"],

    # MetaShift (contexts auto-discovered)
    "metashift": ["train", "val", "test"],

    # OpenMIBOOD (domains auto-discovered per benchmark)
    "openmibood": ["train", "val", "test"],
}

# 

# ============================================
# Detailed Dataset Statistics (Auto-generated)
# ============================================

CLASS_LABELS = {
    # bone_marrow_smears_and_peripheral_blood
    "bone_marrow_smears_and_peripheral_blood": {
        "0": "basophil",
        "1": "blast",
        "2": "erythroblast",
        "3": "eosinophil",
        "4": "lymphocyte_atypical",
        "5": "lymphocyte_typical",
        "6": "metamyelocyte",
        "7": "monocyte",
        "8": "myelocyte",
        "9": "neutrophil_band",
        "10": "neutrophil_segmented",
        "11": "promyelocyte",
        "12": "smudge_cell"
    },
    # retina
    "retina": {
        "0": "no DR",
        "1": "mild",
        "2": "moderate",
        "3": "severe",
        "4": "proliferative DR"
    },
}

SAMPLES_PER_SPLIT = {
    # bone_marrow_smears_and_peripheral_blood
    "bone_marrow_smears_and_peripheral_blood": {
        "train": 147799,
        "val": 18365,
        "test": 41621
    },
    # retina
    "retina": {
        "train": 4862,
        "val": 455,
        "test": 1744
    },
}

INCIDENCE_RATES = {
    # bone_marrow_smears_and_peripheral_blood
    "bone_marrow_smears_and_peripheral_blood": {
        "train": {
            "0": 0.003,
            "1": 0.081,
            "2": 0.2039,
            "3": 0.0399,
            "4": 0.0544,
            "5": 0.1776,
            "6": 0.0207,
            "7": 0.0273,
            "8": 0.0444,
            "9": 0.0674,
            "10": 0.1991,
            "11": 0.0812,
            "12": 0.0003
        },
        "val": {
            "0": 0.0043,
            "1": 0.1794,
            "2": 0.0042,
            "3": 0.0231,
            "4": 0.0006,
            "5": 0.2144,
            "6": 0.0008,
            "7": 0.0974,
            "8": 0.0023,
            "9": 0.0059,
            "10": 0.462,
            "11": 0.0048,
            "12": 0.0008
        },
        "test": {
            "0": 0.0148,
            "1": 0.2068,
            "2": 0.0498,
            "3": 0.0588,
            "4": 0.1678,
            "5": 0.1329,
            "6": 0.0116,
            "7": 0.0603,
            "8": 0.0179,
            "9": 0.0165,
            "10": 0.1723,
            "11": 0.0667,
            "12": 0.0237
        }
    },
    # retina
    "retina": {
        "train": {
            "0": 0.4453,
            "1": 0.1255,
            "2": 0.2548,
            "3": 0.0891,
            "4": 0.0854
        },
        "val": {
            "0": 0.2835,
            "1": 0.0484,
            "2": 0.3429,
            "3": 0.1846,
            "4": 0.1407
        },
        "test": {
            "0": 0.5831,
            "1": 0.1548,
            "2": 0.199,
            "3": 0.043,
            "4": 0.0201
        }
    },
}

SHANNON_EQUITABILITY = {
    # bone_marrow_smears_and_peripheral_blood
    "bone_marrow_smears_and_peripheral_blood": {
        "train": 0.0,
        "val": 0.0,
        "test": 0.0
    },
    # retina
    "retina": {
        "train": 0.8666,
        "val": 0.9064,
        "test": 0.7073
    },
}

NORMALIZATION_MEAN_PER_SPLIT = {
    # bone_marrow_smears_and_peripheral_blood
    "bone_marrow_smears_and_peripheral_blood": {
        "train": [
            0.5531,
            0.4894,
            0.7275
        ],
        "val": [
            0.8209,
            0.7282,
            0.8364
        ],
        "test": [
            0.743,
            0.6563,
            0.7812
        ]
    },
    # retina
    "retina": {
        "train": [
            0.3953,
            0.2224,
            0.0987
        ],
        "val": [
            0.4396,
            0.2131,
            0.0685
        ],
        "test": [
            0.2739,
            0.1268,
            0.0452
        ]
    },
}

NORMALIZATION_STD_PER_SPLIT = {
    # bone_marrow_smears_and_peripheral_blood
    "bone_marrow_smears_and_peripheral_blood": {
        "train": [
            0.2427,
            0.283,
            0.1777
        ],
        "val": [
            0.1712,
            0.2587,
            0.1029
        ],
        "test": [
            0.2027,
            0.2662,
            0.1772
        ]
    },
    # retina
    "retina": {
        "train": [
            0.2806,
            0.1649,
            0.1091
        ],
        "val": [
            0.3097,
            0.1652,
            0.0841
        ],
        "test": [
            0.3133,
            0.1482,
            0.059
        ]
    },
}

IMAGE_DIMENSIONS = {
    # bone_marrow_smears_and_peripheral_blood
    "bone_marrow_smears_and_peripheral_blood": {
        "train": {
            "height": {
                "min": 250,
                "max": 250,
                "avg": 250.0
            },
            "width": {
                "min": 250,
                "max": 250,
                "avg": 250.0
            }
        },
        "val": {
            "height": {
                "min": 400,
                "max": 400,
                "avg": 400.0
            },
            "width": {
                "min": 400,
                "max": 400,
                "avg": 400.0
            }
        },
        "test": {
            "height": {
                "min": 288,
                "max": 288,
                "avg": 288.0
            },
            "width": {
                "min": 288,
                "max": 288,
                "avg": 288.0
            }
        }
    },
    # retina
    "retina": {
        "train": {
            "height": {
                "min": 358,
                "max": 2848,
                "avg": 1613.51
            },
            "width": {
                "min": 474,
                "max": 4288,
                "avg": 1968.92
            }
        },
        "val": {
            "height": {
                "min": 2848,
                "max": 2848,
                "avg": 2848.0
            },
            "width": {
                "min": 4288,
                "max": 4288,
                "avg": 4288.0
            }
        },
        "test": {
            "height": {
                "min": 960,
                "max": 1536,
                "avg": 1345.05
            },
            "width": {
                "min": 1440,
                "max": 2304,
                "avg": 2020.39
            }
        }
    },
}
