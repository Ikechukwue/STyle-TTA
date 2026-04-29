"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for Terra Incognita (wildlife camera trap images from different
locations).

Single-domain domain generalization setup:
    - Train: Stratified training split of the source location (default: location_38).
    - Val: Stratified validation split of the source location.
    - Test: All non-source locations concatenated.
    - Test per location: Individual locations (test_location_43, etc.).

Download & Preparation:
    Download via DomainBed:
        python3 -m domainbed.scripts.download --data_dir=./data

    Or download from Caltech Camera Traps:
        https://lila.science/datasets/caltech-camera-traps/
    Then use the DomainBed preprocessing to extract the 4-location subset.

    The 4 locations used in DomainBed are: L38, L43, L46, L100.
    Each location contains images organized by animal class.

    Directory structure:
        <data_path>/terra_incognita/
        ├── location_38/
        │   ├── bird/
        │   ├── bobcat/
        │   ├── cat/
        │   ├── coyote/
        │   ├── dog/
        │   ├── empty/
        │   ├── opossum/
        │   ├── rabbit/
        │   ├── raccoon/
        │   └── squirrel/
        ├── location_43/
        │   └── (same 10 class folders)
        ├── location_46/
        │   └── (same 10 class folders)
        └── location_100/
            └── (same 10 class folders)

@references:
- Paper: Sara Beery, et al. "Recognition in Terra Incognita." ECCV 2018.
    https://arxiv.org/abs/1807.04975
- DomainBed: Ishaan Gulrajani and David Lopez-Paz. "In Search of Lost Domain
    Generalization." ICLR 2021.
"""

import os
import random
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split


class TerraIncognita(VisionDataset):
    """
    Dataset class for Terra Incognita.

    Domains (locations): location_38, location_43, location_46, location_100
    Classes: 10
    """

    DOMAINS = ['location_38', 'location_43', 'location_46', 'location_100']
    DEFAULT_TRAIN_DOMAIN = 'location_38'

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the Terra Incognita dataset.

        Args:
            root_dir (str): Root directory containing location folders.
            split (str): Dataset split. One of:
                "train", "val", "test", "test_location_38", "test_location_43",
                "test_location_46", "test_location_100".
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                train_domain (str): Source location. Defaults to "location_38".
                size_val_set (float): Val split proportion. Defaults to 0.1.
                seed_train_val_split (int): Seed for train/val split. Defaults to 0.
        """
        super(TerraIncognita, self).__init__(root_dir, transform=transform,
                                             target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

        self.train_domain = kwargs.get('train_domain', self.DEFAULT_TRAIN_DOMAIN)
        assert self.train_domain in self.DOMAINS, \
            f"train_domain must be one of {self.DOMAINS}, got '{self.train_domain}'."

        self.test_domains = [d for d in self.DOMAINS if d != self.train_domain]

        self.dataset = self._load_split(root_dir, split, **kwargs)

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

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
                f"Unknown location '{domain_name}'. Available: {self.DOMAINS}"
            domain_path = os.path.join(root_dir, domain_name)
            self._check_dir(domain_path, domain_name)
            return ImageFolder(domain_path)

        else:
            raise ValueError(f"Invalid split: {split}")

    @staticmethod
    def _check_dir(path: str, name: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Location directory '{name}' not found at {path}. "
                f"Please download Terra Incognita and extract it there."
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
