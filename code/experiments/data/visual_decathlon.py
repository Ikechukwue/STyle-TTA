"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for the Visual Decathlon (VD) challenge.
The Visual Decathlon consists of 10 image classification tasks from different
visual domains. Each task is treated as a separate domain.

Single-domain domain generalization setup:
    - Train: Stratified training split of the source task (default: imagenet12).
    - Val: Stratified validation split of the source task.
    - Test: All non-source tasks concatenated.
    - Test per task: Individual tasks (test_cifar100, test_dtd, etc.).

Download & Preparation:
    Download from: https://www.robots.ox.ac.uk/~vgg/decathlon/
    Download the "decathlon-1.0-data.tar" archive and extract.

    The archive contains 10 datasets in ImageFolder format:
        - imagenet12    (ImageNet subset, 1000 classes)
        - aircraft      (FGVC-Aircraft, 100 classes)
        - cifar100      (CIFAR-100, 100 classes)
        - daimlerpedcls (Daimler Pedestrian, 2 classes)
        - dtd           (Describable Textures, 47 classes)
        - gtsrb         (German Traffic Signs, 43 classes)
        - omniglot      (Omniglot, 1623 classes)
        - svhn          (Street View House Numbers, 10 classes)
        - ucf101        (UCF-101 video frames, 101 classes)
        - vgg-flowers   (VGG Flowers-102, 102 classes)

    Extract into <data_path>/visual_decathlon/ with the following structure.

    Directory structure:
        <data_path>/visual_decathlon/
        ├── imagenet12/
        │   ├── train/
        │   │   └── (class folders)
        │   ├── val/
        │   │   └── (class folders)
        │   └── test/
        │       └── (class folders)
        ├── aircraft/
        │   ├── train/ ...
        │   ├── val/ ...
        │   └── test/ ...
        ├── cifar100/
        │   └── ...
        ├── daimlerpedcls/
        │   └── ...
        ├── dtd/
        │   └── ...
        ├── gtsrb/
        │   └── ...
        ├── omniglot/
        │   └── ...
        ├── svhn/
        │   └── ...
        ├── ucf101/
        │   └── ...
        └── vgg-flowers/
            └── ...

    Note: Each task in the Visual Decathlon comes with its own train/val/test
    folders. For the DG setup, the source task's train folder is split into
    train/val by stratification, and all other tasks' data (train+val+test
    combined) serve as test sets.

@references:
- Paper: Sylvestre-Alvise Rebuffi, et al. "Learning multiple visual domains
    with residual adapters." NeurIPS 2017.
    https://www.robots.ox.ac.uk/~vgg/decathlon/
"""

import os
import random
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split


class VisualDecathlon(VisionDataset):
    """
    Dataset class for the Visual Decathlon challenge.

    Tasks (domains): imagenet12, aircraft, cifar100, daimlerpedcls, dtd,
        gtsrb, omniglot, svhn, ucf101, vgg-flowers
    """

    DOMAINS = [
        'imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd',
        'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers'
    ]
    DEFAULT_TRAIN_DOMAIN = 'imagenet12'

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the Visual Decathlon dataset.

        Args:
            root_dir (str): Root directory containing task folders.
            split (str): Dataset split. One of:
                "train", "val", "test", "test_<task_name>".
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                train_domain (str): Source task. Defaults to "imagenet12".
                size_val_set (float): Val split proportion. Defaults to 0.1.
                seed_train_val_split (int): Seed for train/val split. Defaults to 0.
        """
        super(VisualDecathlon, self).__init__(root_dir, transform=transform,
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
            train_path = os.path.join(root_dir, self.train_domain, 'train')
            self._check_dir(train_path, f"{self.train_domain}/train")
            full_dataset = ImageFolder(train_path)

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
                datasets.extend(self._load_all_splits_for_domain(root_dir, domain))
            return ConcatDataset(datasets)

        elif split.startswith("test_"):
            domain_name = split[len("test_"):]
            assert domain_name in self.DOMAINS, \
                f"Unknown task '{domain_name}'. Available: {self.DOMAINS}"
            sub_datasets = self._load_all_splits_for_domain(root_dir, domain_name)
            if len(sub_datasets) == 1:
                return sub_datasets[0]
            return ConcatDataset(sub_datasets)

        else:
            raise ValueError(f"Invalid split: {split}")

    def _load_all_splits_for_domain(self, root_dir: str, domain: str):
        """Load train+val+test folders for a given domain as a list of datasets."""
        datasets = []
        for sub_split in ['train', 'val', 'test']:
            path = os.path.join(root_dir, domain, sub_split)
            if os.path.isdir(path):
                datasets.append(ImageFolder(path))
        if not datasets:
            # Fallback: domain may be an ImageFolder directly
            path = os.path.join(root_dir, domain)
            self._check_dir(path, domain)
            datasets.append(ImageFolder(path))
        return datasets

    @staticmethod
    def _check_dir(path: str, name: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Directory '{name}' not found at {path}. "
                f"Please download the Visual Decathlon and extract it there."
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
