"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for VLCS (PASCAL VOC 2007, LabelMe, Caltech-101, SUN09).

Single-domain domain generalization setup:
    - Train: Stratified training split of the source domain (default: PASCAL).
    - Val: Stratified validation split of the source domain.
    - Test: All non-source domains concatenated.
    - Test per domain: Individual non-source domains (test_CALTECH, etc.).

Download & Preparation:
    Download via DomainBed:
        python3 -m domainbed.scripts.download --data_dir=./data
    Or download directly from:
        https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8

    Extract into <data_path>/vlcs/ with the following structure.

    Directory structure:
        <data_path>/vlcs/
        ├── CALTECH/
        │   ├── bird/
        │   ├── car/
        │   ├── chair/
        │   ├── dog/
        │   └── person/
        ├── LABELME/
        │   └── (same 5 class folders)
        ├── PASCAL/
        │   └── (same 5 class folders)
        └── SUN/
            └── (same 5 class folders)

@references:
- Paper: Chen Fang, et al. "Unbiased Metric Learning: On the Utilization of
    Multiple Datasets and Web Images for Softening Bias." ICCV 2013.
- DomainBed: Ishaan Gulrajani and David Lopez-Paz. "In Search of Lost Domain
    Generalization." ICLR 2021.
    https://github.com/facebookresearch/DomainBed
"""

import os
import random
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split


class VLCS(VisionDataset):
    """
    Dataset class for VLCS.

    Domains: CALTECH, LABELME, PASCAL, SUN
    Classes: 5 (bird, car, chair, dog, person)
    """

    DOMAINS = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    DEFAULT_TRAIN_DOMAIN = 'PASCAL'

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the VLCS dataset.

        Args:
            root_dir (str): Root directory containing domain folders.
            split (str): Dataset split. One of:
                "train", "val", "test", "test_CALTECH", "test_LABELME",
                "test_PASCAL", "test_SUN".
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                train_domain (str): Source domain. Defaults to "PASCAL".
                size_val_set (float): Val split proportion. Defaults to 0.1.
                seed_train_val_split (int): Seed for train/val split. Defaults to 0.
        """
        super(VLCS, self).__init__(root_dir, transform=transform,
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
                f"Please download VLCS and extract it there."
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
