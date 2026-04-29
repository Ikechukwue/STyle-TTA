"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for Office-Home (4 domains, 65 classes).

Single-domain domain generalization setup:
    - Train: Stratified training split of the source domain (default: Product).
    - Val: Stratified validation split of the source domain.
    - Test: All non-source domains concatenated.
    - Test per domain: Individual non-source domains (test_Art, etc.).

Download & Preparation:
    Download from the official website:
        https://www.hemanthdv.org/officeHomeDataset.html
    Or from the DomainBed download script:
        python3 -m domainbed.scripts.download --data_dir=./data

    Extract into <data_path>/office_home/ with the following structure.

    Directory structure:
        <data_path>/office_home/
        ├── Art/
        │   ├── Alarm_Clock/
        │   ├── Backpack/
        │   └── ... (65 class folders)
        ├── Clipart/
        │   └── ... (65 class folders)
        ├── Product/
        │   └── ... (65 class folders)
        └── Real_World/
            └── ... (65 class folders)

@references:
- Paper: Hemanth Venkateswara, et al. "Deep Hashing Network for Unsupervised
    Domain Adaptation." CVPR 2017.
    https://www.hemanthdv.org/officeHomeDataset.html
"""

import os
import random
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split


class OfficeHome(VisionDataset):
    """
    Dataset class for Office-Home.

    Domains: Art, Clipart, Product, Real_World
    Classes: 65
    """

    DOMAINS = ['Art', 'Clipart', 'Product', 'Real_World']
    DEFAULT_TRAIN_DOMAIN = 'Product'

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the Office-Home dataset.

        Args:
            root_dir (str): Root directory containing domain folders.
            split (str): Dataset split. One of:
                "train", "val", "test", "test_Art", "test_Clipart",
                "test_Product", "test_Real_World".
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                train_domain (str): Source domain. Defaults to "Product".
                size_val_set (float): Val split proportion. Defaults to 0.1.
                seed_train_val_split (int): Seed for train/val split. Defaults to 0.
        """
        super(OfficeHome, self).__init__(root_dir, transform=transform,
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
                f"Please download Office-Home and extract it there."
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
