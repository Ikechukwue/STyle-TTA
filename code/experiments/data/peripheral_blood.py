"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for the Peripheral Blood dataset for domain generalization.
It combines MLL23 (train/val), Matek19 (test), and Acevedo20 (train/val).

@references:
- Acevedo20:
    - Paper: Andrea Acevedo and Santiago Alférez and Anna Merino and Laura Puigví and
        José Rodellar. "Recognition of peripheral blood cell images using convolutional
        neural networks". Computer Methods and Programs in Biomedicine. Volume 180,
        2019.
    - Data: Acevedo, Andrea; Merino, Anna; Alférez, Santiago; Molina, Ángel;
        Boldú, Laura; Rodellar, José (2020), “A dataset for microscopic peripheral
        blood cell images for development of automatic recognition systems".
        Mendeley Data, V1, https://data.mendeley.com/datasets/snkd93bnjr/1
- Matek19:
    - Paper: Matek, C., Schwarz, S., Spiekermann, K.  et al.  Human-level recognition
        of blast cells in acute myeloid leukaemia with convolutional neural networks.
        Nat Mach Intell 1, 538–544 (2019). https://doi.org/10.1038/s42256-019-0101-9
    - Data: Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). A Single-cell
        Morphological Dataset of Leukocytes from AML Patients and Non-malignant
        Controls [Data set].
        The Cancer Imaging Archive. https://doi.org/10.7937/tcia.2019.36f5o9ld“
        https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/
- MLL23:
    - Paper: Shetab Boushehri, S., Kazeminia, S., Gruber, A. et al. A large
        expert-annotated single-cell peripheral blood dataset for hematological disease
        diagnostics. Sci Data 12, 1773 (2025). https://doi.org/10.1038/s41597-025-06223-x
    - Data: Shetab Boushehri, S., Gruber, A., Sun, X., Kazeminia, S., Matek, C.,
        Hehr, M., Spiekermann, K., Pohlkamp, C., Haferlach, T., & Marr, C. (2023).
        A large publicly available single-cell peripheral blood dataset (MLL23)
        [Data set]. Zenodo.
        https://zenodo.org/records/14277609
"""

import os
import shutil
import random
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import ConcatDataset, Subset
from typing import Callable, Optional

from code.experiments.data._utils import create_own_train_val_subset


class PeripheralBlood(VisionDataset):
    """
    Dataset class for the Peripheral Blood dataset for domain generalization.
    We use Matek19 as the test set.
    We use MLL23 and Acevedo20 for training (80%) and validation (20%).
    Only the intersection of classes across all three datasets is used.

    Common Classes:
        - basophil
        - eosinophil
        - erythroblast
        - lymphocyte (atypical)
        - lymphocyte (typical)
        - metamyelocyte
        - monocyte
        - myelocyte
        - myeloblast
        - neutrophil (band)
        - neutrophil (segmented)
        - promyelocyte
        - smudge_cell
    """

    # Intersection of classes
    COMMON_CLASSES = [
        "basophil",
        "eosinophil",
        "erythroblast",
        "lymphocyte_atypical",
        "lymphocyte_typical",
        "metamyelocyte",
        "monocyte",
        "myelocyte",
        "myeloblast",
        "neutrophil_band",
        "neutrophil_segmented",
        "promyelocyte",
        "smudge_cell"
    ]

    # Mapping from dataset-specific class names to common class names
    # If a class is not in this map, it is ignored.
    CLASS_MAPPINGS = {
        "Matek19": {
            # Official assignments from the dataset documentation
            "BAS": "basophil",
            "EBO": "erythroblast",
            "EOS": "eosinophil",
            "KSC": "smudge_cell",
            "LYA": "lymphocyte_atypical",
            "LYT": "lymphocyte_typical",
            "MMZ": "metamyelocyte",
            # "MOB" is ignored (monoblast)
            "MON": "monocyte",
            "MYB": "myelocyte",
            "MYO": "myeloblast",
            "NGB": "neutrophil_band",
            "NGS": "neutrophil_segmented",
            "PMB": "promyelocyte",  # Originally this was "promyelocyte_bilobled" but is collapsed to "promyelocyte"
            "PMO": "promyelocyte",
        },
        "Acevedo20": {
            "basophil": "basophil",
            "eosinophil": "eosinophil",
            "erythroblast": "erythroblast",
            # "immature_granulocyte" ignored
            "lymphocyte": "lymphocyte_typical",  # typical lymphocytes are called just 'lymphocyte' in Acevedo20
            "metamyelocyte": "metamyelocyte",  # Was originally under "immature granulocytes (ig)" but we map it directly
            "monocyte": "monocyte", 
            "myelocyte": "myelocyte",  # Was originally under "immature granulocytes (ig)" but we map it directly
            # "neutrophil" ignored (general category)
            "neutrophil_band": "neutrophil_band",
            "neutrophil_segmented": "neutrophil_segmented",
            # "platelet" ignored
            "promyelocyte": "promyelocyte",  # Was originally under "immature granulocytes (ig)" but we map it directly
        },
        "MLL23": {
            "basophil": "basophil",
            "eosinophil": "eosinophil",
            "hairy_cell": "lymphocyte_atypical",  # Original dataset separated 'atypical lymphocytes' into 'plasma cells', ‘large granular lymphocytes’, ‘reactive lymphocytes’, ‘hairy cells’ and other ‘neoplastic lymphocytes’
            "lymphocyte": "lymphocyte_typical",  # typical lymphocytes are called just 'lymphocyte' in MLL23
            "lymphocyte_large_granular": "lymphocyte_atypical",  # Original dataset separated 'atypical lymphocytes' into 'plasma cells', ‘large granular lymphocytes’, ‘reactive lymphocytes’, ‘hairy cells’ and other ‘neoplastic lymphocytes’
            "lymphocyte_neoplastic": "lymphocyte_atypical",  # Original dataset separated 'atypical lymphocytes' into 'plasma cells', ‘large granular lymphocytes’, ‘reactive lymphocytes’, ‘hairy cells’ and other ‘neoplastic lymphocytes’
            "lymphocyte_reactive": "lymphocyte_atypical",  # Original dataset separated 'atypical lymphocytes' into 'plasma cells', ‘large granular lymphocytes’, ‘reactive lymphocytes’, ‘hairy cells’ and other ‘neoplastic lymphocytes’
            "metamyelocyte": "metamyelocyte",
            "monocyte": "monocyte",
            "myeloblast": "myeloblast",
            "myelocyte": "myelocyte",
            "neutrophil_band": "neutrophil_band",
            "neutrophil_segmented": "neutrophil_segmented",
            "normoblast": "erythroblast",  # 'erythroblast' is called 'normoblast' in MLL23
            "plasma_cell": "lymphocyte_atypical",  # Original dataset separated 'atypical lymphocytes' into 'plasma cells', ‘large granular lymphocytes’, ‘reactive lymphocytes’, ‘hairy cells’ and other ‘neoplastic lymphocytes’
            "promyelocyte": "promyelocyte",
            "promyelocyte_atypical": "promyelocyte",  # Collapsed to "promyelocyte"
            "smudge_cell": "smudge_cell",
        }
    }

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None) -> None:
        super(PeripheralBlood, self).__init__(root_dir, transform=transform,
                                              target_transform=target_transform)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Check if datasets exist
        self._check_datasets()

        # Load data
        if split == 'test':
            self.dataset = self._load_matek19(use_subset, subset_size, subset_seed)
        elif split in ['train', 'val']:
            self.dataset = self._load_train_val(split, use_subset, subset_size,
                                                subset_seed)
        else:
            raise ValueError(f"Invalid split: {split}")

    def _check_datasets(self):
        # MLL23
        mll23_path = os.path.join(self.root_dir, "MLL23")

        # Matek19 & Acevedo20 - Print instructions if missing
        matek19_path = os.path.join(self.root_dir, "Matek19")
        acevedo20_path = os.path.join(self.root_dir, "Acevedo20")
        mll23_path = os.path.join(self.root_dir, "MLL23")

        if not os.path.exists(matek19_path):
            print(f"WARNING: Matek19 dataset not found at {matek19_path}.")
            print("Please download it from https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/ or https://www.kaggle.com/datasets/walkersneps/aml-cytomorphology-lmu")
            print("and extract it so classes are in subfolders.")

        if not os.path.exists(acevedo20_path):
            print(f"WARNING: Acevedo20 dataset not found at {acevedo20_path}.")
            print("Please download it from https://data.mendeley.com/datasets/snkd93bnjr/1")
            print("and extract it so classes are in subfolders.")

        if not os.path.exists(mll23_path):
            print(f"WARNING: MLL23 dataset not found at {mll23_path}.")
            print("Please download it from https://zenodo.org/records/14277609")
            print("and extract it so classes are in subfolders.")

        # Remove __MACOSX if present
        for path in [matek19_path, acevedo20_path, mll23_path]:
            macosx_path = os.path.join(path, "__MACOSX")
            if os.path.exists(macosx_path):
                shutil.rmtree(macosx_path)

    def _load_matek19(self, use_subset, subset_size, subset_seed) -> VisionDataset:
        matek19_path = os.path.join(self.root_dir, "Matek19")
        dataset = ImageFolder(matek19_path)

        # Filter classes
        dataset = self._filter_classes(dataset, "Matek19")

        if use_subset and subset_size is not None:
            dataset = self._apply_subset(dataset, subset_size, subset_seed)

        return dataset

    def _load_train_val(self, split, use_subset, subset_size,
                        subset_seed) -> VisionDataset:
        mll23_path = os.path.join(self.root_dir, "MLL23")
        acevedo20_path = os.path.join(self.root_dir, "Acevedo20")

        datasets = []

        if os.path.exists(mll23_path):
            d_mll = ImageFolder(mll23_path)
            d_mll = self._filter_classes(d_mll, "MLL23")
            # Split 80/20
            d_mll_sub = create_own_train_val_subset(d_mll, split, size_val_set=0.2,
                                                    seed_train_val_split=42)
            datasets.append(d_mll_sub)

        if os.path.exists(acevedo20_path):
            d_acevedo = ImageFolder(acevedo20_path)
            d_acevedo = self._filter_classes(d_acevedo, "Acevedo20")
            # Split 80/20
            d_acevedo_sub = create_own_train_val_subset(d_acevedo, split,
                                                        size_val_set=0.2,
                                                        seed_train_val_split=42)
            datasets.append(d_acevedo_sub)

        if not datasets:
            raise RuntimeError("No training datasets found (MLL23, Acevedo20).")

        concat_dataset = ConcatDataset(datasets)

        if use_subset and subset_size is not None:
            concat_dataset = self._apply_subset(concat_dataset, subset_size,
                                                subset_seed)

        return concat_dataset

    def _filter_classes(self, dataset: ImageFolder, dataset_name: str) -> VisionDataset:
        """
        Filters the ImageFolder dataset to only include common classes
        and remaps indices to a common scheme.
        """
        mapping = self.CLASS_MAPPINGS[dataset_name]
        new_class_to_idx = {c: i for i, c in enumerate(self.COMMON_CLASSES)}

        # Filter samples
        filtered_samples = []

        for path, target in dataset.samples:
            original_class = dataset.classes[target]

            # Check if this class maps to a common class
            if original_class in mapping:
                common_class = mapping[original_class]
                new_target = new_class_to_idx[common_class]
                filtered_samples.append((path, new_target))

        # Create a new dataset with filtered samples
        dataset.samples = filtered_samples
        dataset.targets = [s[1] for s in filtered_samples]
        dataset.classes = self.COMMON_CLASSES
        dataset.class_to_idx = new_class_to_idx

        return dataset

    def _apply_subset(self, dataset, subset_size, subset_seed):
        # Using the same logic as in other datasets
        dataset_size = len(dataset)
        seed = subset_seed if subset_seed is not None else 42
        random.seed(seed)
        torch.manual_seed(seed)

        if subset_size >= dataset_size:
            # Calculate how many times to repeat the dataset
            repeats = (subset_size + dataset_size - 1) // dataset_size
            print(f"Warning: Requested subset size ({subset_size}) is >= dataset size "
                  f"({dataset_size}). Repeating dataset {repeats}x to fulfill "
                  f"requirements.")

            # Create repeated indices with shuffling between repeats
            all_indices = []
            for _ in range(repeats):
                indices = list(range(dataset_size))
                random.shuffle(indices)
                all_indices.extend(indices)

            # Trim to exact subset size
            indices = all_indices[:subset_size]
            self.dataset = Subset(self.dataset, indices)
            print(f"Applied subset: using {subset_size} samples (repeated from "
                  f"{dataset_size} originals, seed: {subset_seed})")
            return

        # Set seed if provided
        if subset_seed is not None:
            random.seed(subset_seed)
            torch.manual_seed(subset_seed)

        # Create random subset
        indices = random.sample(range(dataset_size), subset_size)
        self.dataset = Subset(self.dataset, indices)

        print(f"Applied subset: using {subset_size}/{dataset_size} samples "
              f"(seed: {subset_seed})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target
