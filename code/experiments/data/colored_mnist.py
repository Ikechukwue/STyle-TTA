"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for Colored MNIST from the Invariant Risk Minimization (IRM)
paper. Three environments with different spurious color-label correlations.

Single-domain domain generalization setup:
    - Train: Stratified training split of the source environment (default: env0).
    - Val: Stratified validation split of the source environment.
    - Test: All non-source environments concatenated.
    - Test per env: Individual environments (test_env0, test_env1, test_env2).

Download & Preparation:
    Colored MNIST is generated programmatically from the standard MNIST dataset.
    The generation procedure (from the IRM paper by Arjovsky et al., 2019):

    1. MNIST is auto-downloaded by torchvision.
    2. Labels are binarized: digits 0-4 -> label 0, digits 5-9 -> label 1.
    3. For each environment, the label is flipped with probability p_flip,
       and then the image is colored (red or green) based on the (possibly
       flipped) label with probability p_color:
       - env0 (train): p_flip=0.25, p_color=0.2 (strong correlation)
       - env1 (train): p_flip=0.25, p_color=0.1 (stronger correlation)
       - env2 (test):  p_flip=0.25, p_color=0.9 (anti-correlation)

    The dataset is generated on-the-fly from MNIST.

    Directory structure:
        <data_path>/colored_mnist/
        └── MNIST/               (auto-downloaded by torchvision)
            └── raw/
                └── ...

@references:
- Paper: Martin Arjovsky, et al. "Invariant Risk Minimization." arXiv 2019.
    https://arxiv.org/abs/1907.02893
- Code: https://github.com/facebookresearch/InvariantRiskMinimization
- DomainBed: https://github.com/facebookresearch/DomainBed
"""

import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split


class ColoredMNISTEnv(Dataset):
    """Single environment of Colored MNIST."""

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
        img = Image.fromarray(self.images[idx].astype(np.uint8))
        target = int(self.labels[idx])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ColoredMNIST(VisionDataset):
    """
    Dataset class for Colored MNIST.

    Environments: env0 (p_color=0.2), env1 (p_color=0.1), env2 (p_color=0.9)
    Classes: 2 (binary: digits 0-4 vs 5-9)
    """

    DOMAINS = ['env0', 'env1', 'env2']
    DEFAULT_TRAIN_DOMAIN = 'env0'

    # Environment configurations: (p_flip, p_color)
    ENV_CONFIGS = {
        'env0': (0.25, 0.2),
        'env1': (0.25, 0.1),
        'env2': (0.25, 0.9),
    }

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the Colored MNIST dataset.

        Args:
            root_dir (str): Root directory for MNIST download.
            split (str): Dataset split. One of:
                "train", "val", "test", "test_env0", "test_env1", "test_env2".
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                train_domain (str): Source environment. Defaults to "env0".
                size_val_set (float): Val split proportion. Defaults to 0.1.
                seed_train_val_split (int): Seed for train/val split. Defaults to 0.
                generation_seed (int): Seed for color generation. Defaults to 0.
        """
        super(ColoredMNIST, self).__init__(root_dir, transform=transform,
                                           target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

        self.train_domain = kwargs.get('train_domain', self.DEFAULT_TRAIN_DOMAIN)
        assert self.train_domain in self.DOMAINS, \
            f"train_domain must be one of {self.DOMAINS}, got '{self.train_domain}'."

        self.test_domains = [d for d in self.DOMAINS if d != self.train_domain]
        self.generation_seed = kwargs.get('generation_seed', 0)

        # Generate all environments
        self.envs = self._generate_environments(root_dir)

        self.dataset = self._load_split(split, **kwargs)

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _generate_environments(self, root_dir: str) -> dict:
        """Generate all Colored MNIST environments from MNIST data."""
        mnist_train = MNIST(root=root_dir, train=True, download=True)
        mnist_test = MNIST(root=root_dir, train=False, download=True)

        # Combine train and test, then split into 3 environments
        all_images = torch.cat([mnist_train.data, mnist_test.data])
        all_labels = torch.cat([mnist_train.targets, mnist_test.targets])

        # Binarize labels: 0-4 -> 0, 5-9 -> 1
        binary_labels = (all_labels >= 5).long()

        rng = np.random.RandomState(self.generation_seed)

        # Split into 3 chunks for 3 environments
        n = len(all_images)
        chunk_size = n // 3
        envs = {}
        for i, env_name in enumerate(self.DOMAINS):
            start = i * chunk_size
            end = start + chunk_size if i < 2 else n
            env_images = all_images[start:end].numpy()
            env_labels = binary_labels[start:end].numpy()

            p_flip, p_color = self.ENV_CONFIGS[env_name]

            # Flip labels with probability p_flip
            flip_mask = rng.random(len(env_labels)) < p_flip
            env_labels = np.where(flip_mask, 1 - env_labels, env_labels)

            # Assign colors based on label with noise p_color
            color_mask = rng.random(len(env_labels)) < p_color
            colors = np.where(color_mask, 1 - env_labels, env_labels)

            # Create colored images (28x28 -> 28x28x3)
            colored_images = np.stack([env_images, env_images, np.zeros_like(env_images)],
                                      axis=-1)
            for j in range(len(colored_images)):
                if colors[j] == 1:
                    # Green channel
                    colored_images[j, :, :, 0] = 0
                    colored_images[j, :, :, 1] = env_images[j]
                    colored_images[j, :, :, 2] = 0
                else:
                    # Red channel
                    colored_images[j, :, :, 0] = env_images[j]
                    colored_images[j, :, :, 1] = 0
                    colored_images[j, :, :, 2] = 0

            # Restore original (unflipped) labels for evaluation
            original_labels = binary_labels[start:end].numpy()
            envs[env_name] = ColoredMNISTEnv(colored_images, original_labels)

        return envs

    def _load_split(self, split: str, **kwargs):
        size_val_set = kwargs.get('size_val_set', 0.1)
        seed_split = kwargs.get('seed_train_val_split', 0)

        if split in ["train", "val"]:
            full_dataset = self.envs[self.train_domain]
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
            datasets = [self.envs[d] for d in self.test_domains]
            return ConcatDataset(datasets)

        elif split.startswith("test_"):
            env_name = split[len("test_"):]
            assert env_name in self.DOMAINS, \
                f"Unknown environment '{env_name}'. Available: {self.DOMAINS}"
            return self.envs[env_name]

        else:
            raise ValueError(f"Invalid split: {split}")

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
