"""
xAILab Bamberg
University of Bamberg

@description:
Dataset classes for CIFAR-10, CIFAR-100, CIFAR-10-C, and CIFAR-100-C.

Single-domain domain generalization setup:
    - Train: Stratified training split of clean CIFAR training set
    - Val: Stratified validation split of clean CIFAR training set
    - Test: Clean CIFAR test set
    - Test-C: CIFAR-C corrupted test sets (15 corruption types x 5 severity levels)

Download & Preparation:
    1. CIFAR-10 and CIFAR-100 are auto-downloaded by torchvision.

    2. CIFAR-10-C:
       Download from https://zenodo.org/record/2535967
       Extract into: <root_dir>/CIFAR-10-C/
       Expected files: 15 corruption .npy files + labels.npy

    3. CIFAR-100-C:
       Download from https://zenodo.org/record/3555552
       Extract into: <root_dir>/CIFAR-100-C/
       Expected files: 15 corruption .npy files + labels.npy

    Directory structure:
        <data_path>/cifar/
        ├── cifar-10-batches-py/    (auto-downloaded by torchvision)
        ├── cifar-100-python/       (auto-downloaded by torchvision)
        ├── CIFAR-10-C/
        │   ├── brightness.npy
        │   ├── contrast.npy
        │   ├── defocus_blur.npy
        │   ├── elastic_transform.npy
        │   ├── fog.npy
        │   ├── frost.npy
        │   ├── gaussian_noise.npy
        │   ├── glass_blur.npy
        │   ├── impulse_noise.npy
        │   ├── jpeg_compression.npy
        │   ├── motion_blur.npy
        │   ├── pixelate.npy
        │   ├── shot_noise.npy
        │   ├── snow.npy
        │   ├── zoom_blur.npy
        │   └── labels.npy
        └── CIFAR-100-C/
            ├── (same .npy files as above)
            └── labels.npy

@references:
- CIFAR-10/100:
    Alex Krizhevsky. "Learning Multiple Layers of Features from Tiny Images." 2009.
    https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-C:
    Dan Hendrycks and Thomas Dietterich. "Benchmarking Neural Network Robustness
    to Common Corruptions and Perturbations." ICLR 2019.
    https://github.com/hendrycks/robustness
"""

import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split


# Standard 15 corruption types used for mCE computation
CIFAR_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

CIFAR_C_SEVERITIES = [1, 2, 3, 4, 5]


class CIFARCorruptedDataset(Dataset):
    """Wrapper for loading CIFAR-C numpy arrays as a PyTorch Dataset."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = Image.fromarray(self.images[idx])
        target = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR(VisionDataset):
    """
    Dataset class for CIFAR-10 and CIFAR-100 with domain generalization setup.

    Splits:
        - "train": Stratified training portion of the clean CIFAR training set.
        - "val": Stratified validation portion of the clean CIFAR training set.
        - "test": Clean CIFAR test set.
        - "test_c": CIFAR-C corrupted test set. Use kwargs to specify
          corruption_type and severity.
    """

    def __init__(self, root_dir: str, split: str, num_classes: int = 10,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the CIFAR dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            split (str): Dataset split ("train", "val", "test", "test_c").
            num_classes (int): 10 for CIFAR-10, 100 for CIFAR-100. Defaults to 10.
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                size_val_set (float): Proportion of training set for validation.
                    Defaults to 0.1.
                seed_train_val_split (int): Seed for train/val split.
                    Defaults to 0.
                corruption_type (str): Corruption name for test_c split.
                    If None, uses all corruptions.
                severity (int): Severity level 1-5 for test_c split.
                    If None, uses all severities.
        """
        super(CIFAR, self).__init__(root_dir, transform=transform,
                                    target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes

        assert split in ["train", "val", "test", "test_c"], \
            f"Split must be 'train', 'val', 'test', or 'test_c', got '{split}'."
        assert num_classes in [10, 100], \
            f"num_classes must be 10 or 100, got {num_classes}."

        self.dataset = self._load_split(root_dir, split, **kwargs)

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _load_split(self, root_dir: str, split: str, **kwargs) -> Dataset:
        size_val_set = kwargs.get('size_val_set', 0.1)
        seed_split = kwargs.get('seed_train_val_split', 0)

        CIFARClass = CIFAR10 if self.num_classes == 10 else CIFAR100

        if split in ["train", "val"]:
            full_train = CIFARClass(root=root_dir, train=True, download=True)
            labels = [full_train[i][1] for i in range(len(full_train))]
            train_idx, val_idx = train_test_split(
                range(len(full_train)),
                test_size=size_val_set,
                random_state=seed_split,
                stratify=labels,
            )
            if split == "train":
                return Subset(full_train, train_idx)
            else:
                return Subset(full_train, val_idx)

        elif split == "test":
            return CIFARClass(root=root_dir, train=False, download=True)

        elif split == "test_c":
            return self._load_cifar_c(root_dir, **kwargs)

    def _load_cifar_c(self, root_dir: str, **kwargs) -> Dataset:
        corruption_type = kwargs.get('corruption_type', None)
        severity = kwargs.get('severity', None)

        prefix = "CIFAR-10-C" if self.num_classes == 10 else "CIFAR-100-C"
        c_dir = os.path.join(root_dir, prefix)

        if not os.path.isdir(c_dir):
            raise FileNotFoundError(
                f"CIFAR-C directory not found at {c_dir}. "
                f"Download from https://zenodo.org/record/2535967 (CIFAR-10-C) or "
                f"https://zenodo.org/record/3555552 (CIFAR-100-C) and extract there."
            )

        labels_all = np.load(os.path.join(c_dir, 'labels.npy'))

        corruptions = [corruption_type] if corruption_type else CIFAR_C_CORRUPTIONS
        severities = [severity] if severity else CIFAR_C_SEVERITIES

        datasets = []
        for corr in corruptions:
            corr_path = os.path.join(c_dir, f'{corr}.npy')
            if not os.path.isfile(corr_path):
                raise FileNotFoundError(f"Corruption file not found: {corr_path}")
            images_all = np.load(corr_path)

            for sev in severities:
                start = (sev - 1) * 10000
                end = sev * 10000
                images = images_all[start:end]
                labels = labels_all[start:end]
                datasets.append(CIFARCorruptedDataset(images, labels))

        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

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
