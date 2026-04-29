"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for OpenMIBOOD (Open Medical Imaging Benchmarks for
Out-of-Distribution Detection). Contains three medical imaging benchmarks with
covariate-shifted in-distribution (cs-ID), near-OOD, and far-OOD splits.

Benchmarks:
    - MIDOG22: Histopathology mitosis detection across scanners.
    - PhaKIR: Surgical phase recognition across procedure types.
    - OASIS: Brain MRI analysis across scanners/protocols.

Single-domain domain generalization setup:
    - Train: Stratified training split of the ID (in-distribution) domain.
    - Val: Stratified validation split of the ID domain.
    - Test: All non-ID domains concatenated (cs-ID + near-OOD + far-OOD).
    - Test per domain: Individual domain splits.

Download & Preparation:
    OpenMIBOOD:
        1. Clone the repository:
            git clone https://github.com/remic-othr/OpenMIBOOD.git
        2. Follow the dataset-specific download instructions in the repository.
        3. Organize data into ImageFolder format per benchmark.

    Directory structure (per benchmark):
        <data_path>/openmibood/<benchmark_name>/
        ├── id/
        │   ├── <class_1>/
        │   │   ├── image1.png
        │   │   └── ...
        │   └── <class_2>/
        │       └── ...
        ├── cs_id/
        │   └── ... (class folders)
        ├── near_ood/
        │   └── ... (class folders)
        └── far_ood/
            └── ... (class folders)

    Example for MIDOG22 (scanners as domains):
        <data_path>/openmibood/midog22/
        ├── XR/
        │   ├── mitosis/
        │   └── non_mitosis/
        ├── S360/
        │   └── ...
        ├── CS/
        │   └── ...
        └── GT450/
            └── ...

    Example for PhaKIR:
        <data_path>/openmibood/phakir/
        ├── cholecystectomy/
        │   ├── phase_0/
        │   └── ...
        ├── appendectomy/
        │   └── ...
        └── hernia/
            └── ...

    The exact directory structure depends on how you organize the downloaded data.
    The dataset auto-discovers domain folders from the root directory. Use the
    'train_domain' kwarg to select which domain serves as the source.

@references:
- OpenMIBOOD: P. Remic et al. "OpenMIBOOD: Open Medical Imaging Benchmarks
    for Out-of-Distribution Detection." 2024.
    https://github.com/remic-othr/OpenMIBOOD
"""

import os
import random
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, List, Optional, Tuple
from sklearn.model_selection import train_test_split


class OpenMIBOOD(VisionDataset):
    """
    Dataset class for OpenMIBOOD medical imaging benchmarks.

    Domains are auto-discovered from the directory structure.
    The first domain alphabetically is used as the default training domain
    (typically 'id' if following the recommended structure).
    """

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the OpenMIBOOD dataset.

        Args:
            root_dir (str): Root directory containing domain folders for one
                benchmark (e.g., .../openmibood/midog22/).
            split (str): Dataset split. One of:
                "train", "val", "test", "test_<domain_name>".
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                train_domain (str): Source domain name. Defaults to first
                    alphabetically (e.g., 'id').
                domains (list): Explicit list of domain names to use.
                    If not provided, auto-discovered from root_dir.
                size_val_set (float): Val split proportion. Defaults to 0.1.
                seed_train_val_split (int): Seed for train/val split. Defaults to 0.
        """
        super(OpenMIBOOD, self).__init__(root_dir, transform=transform,
                                         target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

        # Auto-discover or use provided domains
        explicit_domains = kwargs.get('domains', None)
        if explicit_domains:
            self.DOMAINS = list(explicit_domains)
        else:
            self.DOMAINS = self._discover_domains(root_dir)

        assert len(self.DOMAINS) >= 2, \
            f"Need at least 2 domains, found {len(self.DOMAINS)} in {root_dir}."

        self.train_domain = kwargs.get('train_domain', self.DOMAINS[0])
        assert self.train_domain in self.DOMAINS, \
            f"train_domain must be one of {self.DOMAINS}, got '{self.train_domain}'."

        self.test_domains = [d for d in self.DOMAINS if d != self.train_domain]

        self.dataset = self._load_split(root_dir, split, **kwargs)

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    @staticmethod
    def _discover_domains(root_dir: str) -> List[str]:
        """Auto-discover domain folders from root directory."""
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        domains = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
            and not d.startswith('.')
        ])
        return domains

    def _load_split(self, root_dir: str, split: str, **kwargs):
        size_val_set = kwargs.get('size_val_set', 0.1)
        seed_split = kwargs.get('seed_train_val_split', 0)

        if split in ["train", "val"]:
            domain_path = os.path.join(root_dir, self.train_domain)
            self._check_dir(domain_path, self.train_domain)
            full_dataset = ImageFolder(domain_path)

            labels = [full_dataset[i][1] for i in range(len(full_dataset))]
            train_idx, val_idx = train_test_split(
                range(len(full_dataset)),
                test_size=size_val_set,
                random_state=seed_split,
                stratify=labels,
            )
            if split == "train":
                return Subset(full_dataset, train_idx)
            else:
                return Subset(full_dataset, val_idx)

        elif split == "test":
            datasets = []
            for domain in self.test_domains:
                domain_path = os.path.join(root_dir, domain)
                self._check_dir(domain_path, domain)
                datasets.append(ImageFolder(domain_path))
            return ConcatDataset(datasets)

        elif split.startswith("test_"):
            domain_name = split[len("test_"):]
            assert domain_name in self.DOMAINS, \
                f"Unknown domain '{domain_name}'. Available: {self.DOMAINS}"
            domain_path = os.path.join(root_dir, domain_name)
            self._check_dir(domain_path, domain_name)
            return ImageFolder(domain_path)

        else:
            raise ValueError(f"Invalid split: {split}")

    @staticmethod
    def _check_dir(path: str, name: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Domain directory '{name}' not found at {path}. "
                f"Please download the OpenMIBOOD data following the "
                f"repository instructions: "
                f"https://github.com/remic-othr/OpenMIBOOD"
            )

    def _apply_subset(self, subset_size: int,
                      subset_seed: Optional[int] = None) -> None:
        dataset_size = len(self.dataset)
        seed = subset_seed if subset_seed is not None else 42
        random.seed(seed)

        if subset_size >= dataset_size:
            repeats = (subset_size + dataset_size - 1) // dataset_size
            print(f"Warning: Requested subset size ({subset_size}) is >= dataset size "
                  f"({dataset_size}). Repeating dataset {repeats}x.")
            all_indices = []
            for _ in range(repeats):
                indices = list(range(dataset_size))
                random.shuffle(indices)
                all_indices.extend(indices)
            indices = all_indices[:subset_size]
            self.dataset = Subset(self.dataset, indices)
            return

        indices = random.sample(range(dataset_size), subset_size)
        self.dataset = Subset(self.dataset, indices)
        print(f"Using subset of {subset_size} samples from {dataset_size} "
              f"total samples (seed={seed}).")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
