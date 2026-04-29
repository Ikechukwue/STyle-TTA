"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for the Retina dataset for domain generalization.
It combines APTOS and DeepDR (train), IDRiD (val), and MESSIDOR-2 (test).

@references:
- APTOS:
    - Data: https://www.kaggle.com/c/aptos2019-blindness-detection
- DeepDR:
    - Data: https://isbi.deepdr.org
- IDRiD:
    - Data: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- MESSIDOR-2:
    - Data: https://www.adcis.net/en/third-party/messidor2/
"""

import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class Retina(VisionDataset):
    """
    Dataset class for the Retina dataset for domain generalization.
    We use APTOS + DeepDR for training.
    We use IDRiD for validation.
    We use MESSIDOR-2 for testing.

    Classes (Diabetic Retinopathy grades):
        - 0: No DR
        - 1: Mild
        - 2: Moderate
        - 3: Severe
        - 4: Proliferative DR
    """

    # Split-to-subdataset mapping
    SPLIT_DATASETS = {
        'train': ['aptos', 'deepdr'],
        'val': ['idrid'],
        'test': ['messidor'],
    }

    # Domain IDs for each subdataset
    DATASET_IDS = {'aptos': 0, 'deepdr': 1, 'idrid': 2, 'messidor': 3}

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None) -> None:
        """
        Initialize the Retina dataset.

        Args:
            root_dir (str): Root directory of the dataset.
            split (str): Split of the dataset (train, val, test).
            transform (callable, optional): Transformations to apply to
                the images. Defaults to None.
            target_transform (callable, optional): Transformations to apply
                to the target. Defaults to None.
            use_subset (bool): Whether to use only a subset of the dataset.
                Defaults to False.
            subset_size (int, optional): Number of samples to use if use_subset
                is True. Defaults to None.
            subset_seed (int, optional): Random seed for subset selection.
                Defaults to None.
        """
        super(Retina, self).__init__(root_dir, transform=transform,
                                     target_transform=target_transform)
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Validate split
        if split not in self.SPLIT_DATASETS:
            raise ValueError(f"Invalid split: {split}. Must be one of "
                             f"{list(self.SPLIT_DATASETS.keys())}.")

        # Check if datasets exist
        self._check_datasets()

        # Initialize containers
        self.fpaths, self.labels, self.domains = [], [], []

        # Load data based on selected split
        self._load_dataset()

        # Apply subsetting if requested
        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _check_datasets(self):
        """Check if expected dataset directories exist and print warnings if missing."""
        dataset_info = {
            'aptos': {
                'path': os.path.join(self.root_dir, 'aptos'),
                'url': 'https://www.kaggle.com/c/aptos2019-blindness-detection',
            },
            'deepdr': {
                'path': os.path.join(self.root_dir, 'deepdr'),
                'url': 'https://isbi.deepdr.org',
            },
            'idrid': {
                'path': os.path.join(self.root_dir, 'idrid'),
                'url': 'https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid',
            },
            'messidor': {
                'path': os.path.join(self.root_dir, 'messidor'),
                'url': 'https://www.adcis.net/en/third-party/messidor2/',
            },
        }

        for name in self.SPLIT_DATASETS[self.split]:
            info = dataset_info[name]
            if not os.path.exists(info['path']):
                print(f"WARNING: {name} dataset not found at {info['path']}.")
                print(f"Please download it from {info['url']}.")

    def _load_dataset(self):
        """Load all subdatasets for the current split."""
        datasets = self.SPLIT_DATASETS[self.split]

        for dataset_str in datasets:
            if dataset_str == 'aptos':
                self._load_aptos()
            elif dataset_str == 'deepdr':
                self._load_deepdr()
            elif dataset_str == 'idrid':
                self._load_idrid()
            elif dataset_str == 'messidor':
                self._load_messidor()

    def _apply_subset(self, subset_size: int,
                      subset_seed: Optional[int] = None) -> None:
        """
        Apply subsetting to the dataset.

        Args:
            subset_size (int): Number of samples to use for the subset.
            subset_seed (int, optional): Random seed for subset selection.
                If None, uses default seed 42.
        """
        dataset_size = len(self.fpaths)
        seed = subset_seed if subset_seed is not None else 42
        random.seed(seed)

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
        else:
            # Create random indices for the subset
            indices = random.sample(range(dataset_size), subset_size)

        # Apply subset to all containers
        self.fpaths = [self.fpaths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.domains = [self.domains[i] for i in indices]

        print(f"Using subset of {subset_size} samples from "
              f"{dataset_size} total samples (seed={subset_seed}).")

    def __len__(self) -> int:
        return len(self.fpaths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = Image.open(self.fpaths[idx]).convert("RGB")
        target = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_aptos(self):
        """
        Loads the entire APTOS dataset (train + val + test) into one dataset.
        """
        # Define all splits
        splits_info = [
            {
                'csv': self.root_dir / 'aptos' / 'train_1.csv',
                'img_dir': self.root_dir / 'aptos' / 'train_images' / 'train_images'
            },
            {
                'csv': self.root_dir / 'aptos' / 'valid.csv',
                'img_dir': self.root_dir / 'aptos' / 'val_images' / 'val_images'
            },
            {
                'csv': self.root_dir / 'aptos' / 'test.csv',
                'img_dir': self.root_dir / 'aptos' / 'test_images' / 'test_images'
            }
        ]

        # Iterate over each split, load data
        for split in splits_info:
            csv_path = split['csv']
            img_dir = split['img_dir']

            # Load the CSV
            df = pd.read_csv(csv_path)

            # Create file paths, labels, domains
            for _, row in df.iterrows():
                img_file = row['id_code']
                label = row['diagnosis']

                fpath = img_dir / f'{img_file}.png'

                if not Path.is_file(fpath):
                    print(f"Not found, skipped: {fpath}")
                    continue

                self.fpaths.append(fpath)
                self.labels.append(label)
                self.domains.append(self.DATASET_IDS['aptos'])

    def _load_deepdr(self):
        """
        Loads the DeepDR dataset using 'patient_DR_Level' as the label.
        """
        csv_path = self.root_dir / 'deepdr' / 'regular-fundus-training.csv'

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_rel_path = row['image_path'].replace("\\", "/")
            fpath = self.root_dir / 'deepdr' / img_rel_path[1:]

            label = row['patient_DR_Level']

            if np.isnan(label):
                print(f"NaN label, skipped: {fpath}")
                continue

            label = int(label)

            if not Path.is_file(fpath):
                print(f"Not found, skipped: {fpath}")
                continue

            self.fpaths.append(fpath)
            self.labels.append(label)
            self.domains.append(self.DATASET_IDS['deepdr'])

    def _load_idrid(self):
        """
        Loads the IDRiD dataset.
        """
        csv_path = self.root_dir / 'idrid' / 'idrid_labels.csv'
        img_dir = self.root_dir / 'idrid' / 'Imagenes' / 'Imagenes'

        # Load CSV
        df = pd.read_csv(csv_path)

        # Use only id_code and diagnosis
        for _, row in df.iterrows():
            img_file = row['id_code'] + ".jpg"  # files are IDRiD_001.jpg etc
            label = row['diagnosis']
            fpath = img_dir / img_file

            if not Path.is_file(fpath):
                print(f"Not found, skipped: {fpath}")
                continue

            self.fpaths.append(fpath)
            self.labels.append(label)
            self.domains.append(self.DATASET_IDS['idrid'])

    def _load_messidor(self):
        """
        Loads the Messidor-2 dataset.

        4 Images do not contain label:
        Nan label `(nan)`, skipped: 20060411_58550_0200_PP.png
        Nan label `(nan)`, skipped: IM002385.JPG
        Nan label `(nan)`, skipped: IMAGES/IM003718.JPG
        Nan label `(nan)`, skipped: M004176.JPG
        """
        csv_path = self.root_dir / 'messidor' / 'messidor_data.csv'
        img_dir = self.root_dir / 'messidor' / 'IMAGES'

        # Load CSV
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            img_file = row['image_id']  # already has .png
            label = row['adjudicated_dr_grade']
            fpath = img_dir / img_file.replace(".jpg", ".JPG")  # csv has .jpg but files are uppercase

            if np.isnan(label):
                print(f"Nan label `({label})`, skipped: {fpath}")
                continue

            label = int(label)

            if not Path.is_file(fpath):
                print(f"Not found, skipped: {fpath}")
                continue

            if label not in {0, 1, 2, 3, 4}:
                print(f"Wrong label `({label})`, skipped: {fpath}")
                continue

            self.fpaths.append(fpath)
            self.labels.append(label)
            self.domains.append(self.DATASET_IDS['messidor'])
