"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for the Bone Marrow Smears and Peripheral Blood dataset for domain
generalization.
It combines BMC (train), Matek19 (val), and MLL23 (test).

@references:
- BMC:
    - Paper: Matek, C., Krappe, S., Münzenmayer, C., Haferlach, T., and Marr, C. (2021).
        Highly accurate differentiation of bone marrow cell morphologies using deep
        neural networks on a large image dataset.
    - Data: Matek, C., Krappe, S., Münzenmayer, C., Haferlach, T., & Marr, C. (2021).
        An Expert-Annotated Dataset of Bone Marrow Cytology in Hematologic Malignancies
        [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.AXH3-T579
        https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/
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
import random
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Subset
from typing import Callable, Optional


class BoneMarrowSmearsAndPeripheralBlood(VisionDataset):
    """
    Dataset class for the Bone Marrow Smears and Peripheral Blood dataset for domain
    generalization.
    It combines three datasets: BMC (train), Matek19 (val), and MLL23 (test).
    Only the intersection of classes across all three datasets is used.
    Due to different levels of granularity in class definitions, some classes are
    collapsed or ignored to form a common set of classes.
    """

    # Broken samples (identified by index in the original dataset) that should be skipped
    BROKEN_SAMPLES = {
        "BMC": [85014],  # Example indices of broken samples in BMC
        "Matek19": [],  # Example indices of broken samples in Matek19
        "MLL23": []  # Example indices of broken samples in MLL23
    }

    # Intersection of classes
    COMMON_CLASSES = [
        "basophil",
        "blast",
        "erythroblast",
        "eosinophil",
        "lymphocyte_atypical",
        "lymphocyte_typical",
        "metamyelocyte",
        "monocyte",
        "myelocyte",
        "neutrophil_band",
        "neutrophil_segmented",
        "promyelocyte",
        "smudge_cell"
    ]

    # Mapping from dataset-specific class names to common class names
    # If a class is not in this map, it is ignored.
    CLASS_MAPPINGS = {
        "BMC": {
            # Collapsed / remapped classes to match common classes
            # - ABE: "abnormal eosinophil" collapsed to "eosinophil"
            # - ART: "artefact" ignored
            # - FGC: "faggott_cell" ignored
            # - HAC: "hairy_cell" collapsed to "lymphocyte_atypical"
            # - LYI: "lymphocyte_immature" ignored
            # - LYT: "lymphocyte" renamed to "lymphocyte_typical"
            # - NIF: "not_identifiable" ignored
            # - OTH: "other_cell" ignored
            # - PEB: "proerythroblast" collapsed to "erythroblast"
            # - PLM: "plasma_cell" collapsed to "lymphocyte_atypical"
            "ABE": "eosinophil",
            # "ART": "artefact",
            "BAS": "basophil",
            "BLA": "blast",
            "EBO": "erythroblast",
            "EOS": "eosinophil",
            # "FGC": "faggott_cell",
            "HAC": "lymphocyte_atypical",
            "KSC": "smudge_cell",
            # "LYI": "lymphocyte_immature",
            "LYT": "lymphocyte_typical",
            "MMZ": "metamyelocyte",
            "MON": "monocyte",
            "MYB": "myelocyte",
            "NGB": "neutrophil_band",
            "NGS": "neutrophil_segmented",
            # "NIF": "not_identifiable",
            # "OTH": "other_cell",
            "PEB": "erythroblast",
            "PLM": "lymphocyte_atypical",
            "PMO": "promyelocyte"
        },
        "Matek19": {
            # Collapsed / remapped classes to match common classes
            # - MOB: "monoblast" collapsed to "blast"
            # - MYO: "myeloblast" collapsed to "blast"
            # - PMB: "promyelocyte_bilobled" collapsed to "promyelocyte"
            "BAS": "basophil",
            "EBO": "erythroblast",
            "EOS": "eosinophil",
            "KSC": "smudge_cell",
            "LYA": "lymphocyte_atypical",
            "LYT": "lymphocyte_typical",
            "MMZ": "metamyelocyte",
            "MOB": "blast",
            "MON": "monocyte",
            "MYB": "myelocyte",
            "MYO": "blast",
            "NGB": "neutrophil_band",
            "NGS": "neutrophil_segmented",
            "PMB": "promyelocyte",
            "PMO": "promyelocyte",
        },
        "MLL23": {
            # Collapsed / remapped classes to match common classes
            # - 'plasma cells', ‘large granular lymphocytes’, ‘reactive lymphocytes’,
            #   ‘hairy cells’, and ‘neoplastic lymphocytes’ collapsed
            #   to atypical lymphocytes
            # - 'lymphocyte' renamed to 'lymphocyte_typical'
            # - 'normoblast' renamed to 'erythroblast'
            # - 'lymphocyte' renamed to 'lymphocyte_typical'
            # - 'myeloblast' renamed to 'blast'
            # - 'promyelocyte_atypical' collapsed to 'promyelocyte'
            "basophil": "basophil",
            "eosinophil": "eosinophil",
            "hairy_cell": "lymphocyte_atypical",
            "lymphocyte": "lymphocyte_typical",
            "lymphocyte_large_granular": "lymphocyte_atypical",
            "lymphocyte_neoplastic": "lymphocyte_atypical",
            "lymphocyte_reactive": "lymphocyte_atypical",
            "metamyelocyte": "metamyelocyte",
            "monocyte": "monocyte",
            "myeloblast": "blast",
            "myelocyte": "myelocyte",
            "neutrophil_band": "neutrophil_band",
            "neutrophil_segmented": "neutrophil_segmented",
            "normoblast": "erythroblast",
            "plasma_cell": "lymphocyte_atypical",
            "promyelocyte": "promyelocyte",
            "promyelocyte_atypical": "promyelocyte",
            "smudge_cell": "smudge_cell",
        }
    }

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None) -> None:
        super(BoneMarrowSmearsAndPeripheralBlood, self).__init__(root_dir,
                                                                 transform=transform,
                                                                 target_transform=target_transform)
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Check if datasets exist
        self._check_datasets()

        # Load data
        self.dataset = self._load_split(split, use_subset, subset_size, subset_seed)
        # if split == 'train':
        #     self.dataset = self._load_bmc(use_subset, subset_size, subset_seed)
        # elif split == 'val':
        #     self.dataset = self._load_matek19(use_subset, subset_size, subset_seed)
        # elif split == 'test':
        #     self.dataset = self._load_mll23(use_subset, subset_size, subset_seed)
        # else:
        #     raise ValueError(f"Invalid split: {split}")

    def _check_datasets(self):
        # Print instructions if missing
        bmc_path = os.path.join(self.root_dir, "BMC")
        matek19_path = os.path.join(self.root_dir, "Matek19")
        mll23_path = os.path.join(self.root_dir, "MLL23")

        if not os.path.exists(bmc_path):
            print(f"WARNING: BMC dataset not found at {bmc_path}.")
            print("Please download it from https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/")
            print("and extract it so classes are in subfolders.")

        if not os.path.exists(matek19_path):
            print(f"WARNING: Matek19 dataset not found at {matek19_path}.")
            print("Please download it from https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/")
            print("and extract it so classes are in subfolders.")

        if not os.path.exists(mll23_path):
            print(f"WARNING: MLL23 dataset not found at {mll23_path}.")
            print("Please download it from https://zenodo.org/records/14277609")
            print("and extract it so classes are in subfolders.")

    def _load_split(self, split: str, use_subset, subset_size,
                    subset_seed) -> VisionDataset:
        # Determine dataset name based on split
        if split == 'train':
            dataset_name = "BMC"
        elif split == 'val':
            dataset_name = "Matek19"
        elif split == 'test':
            dataset_name = "MLL23"
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load dataset
        dataset_path = os.path.join(self.root_dir, dataset_name)
        dataset = ImageFolder(dataset_path)

        # Filter classes
        dataset = self._filter_classes(dataset, dataset_name)

        # Remove broken samples
        broken_indices = self.BROKEN_SAMPLES.get(dataset_name, [])
        if broken_indices:
            dataset.samples = [s for i, s in enumerate(dataset.samples) if i not in broken_indices]
            dataset.targets = [s[1] for s in dataset.samples]
            print(f"Removed {len(broken_indices)} broken samples from {dataset_name}.")

        # Apply subset if requested
        if use_subset and subset_size is not None:
            dataset = self._apply_subset(dataset, subset_size, subset_seed)

        return dataset

    def _filter_classes(self, dataset: ImageFolder, dataset_name: str) -> VisionDataset:
        """
        Filters the ImageFolder dataset to only include common classes
        and remaps indices to a common scheme.
        """
        mapping = self.CLASS_MAPPINGS[dataset_name]

        # Identify valid indices
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

        # Update dataset samples and targets
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
        # Try to load the image, handle corrupted files
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                img, target = self.dataset[idx]
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    target = self.target_transform(target)
                return img, target
            except (OSError, IOError) as e:
                # Corrupted image file, try next index
                print(f"Warning: Corrupted image at index {idx}, skipping. Error: {e}")
                idx = (idx + 1) % len(self.dataset)
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to load valid image after {max_attempts} attempts")
