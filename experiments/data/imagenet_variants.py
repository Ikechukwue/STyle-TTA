"""
xAILab Bamberg
University of Bamberg

@description:
Dataset classes for ImageNet-1k and its distribution-shifted variants:
ImageNet-A, ImageNet-R, ImageNet-C, ImageNet-P, ImageNet-Sketch, ImageNet-V2.

Single-domain domain generalization setup:
    - Train: ImageNet-1k training set
    - Val: ImageNet-1k validation set
    - Test: ImageNet-1k validation set (as in-distribution test)
    - Test variants: ImageNet-A, ImageNet-R, ImageNet-C, ImageNet-P,
      ImageNet-Sketch, ImageNet-V2

Download & Preparation:
    1. ImageNet-1k (ILSVRC2012):
       Request access at https://image-net.org/ and download:
       - ILSVRC2012_img_train.tar -> extract to <root_dir>/imagenet1k/train/
       - ILSVRC2012_img_val.tar -> extract to <root_dir>/imagenet1k/val/
       Then run the official script to reorganize val into per-class folders:
       https://raw.githubusercontent.com/soumith/imagenetloader/master/valprep.sh

    2. ImageNet-A:
       wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
       Extract to <root_dir>/imagenet-a/
       Contains 7,500 images across 200 ImageNet classes.

    3. ImageNet-R (Renditions):
       wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
       Extract to <root_dir>/imagenet-r/
       Contains 30,000 renditions across 200 ImageNet classes.

    4. ImageNet-C:
       Download from https://zenodo.org/record/2235448
       Extract to <root_dir>/imagenet-c/
       Structure: imagenet-c/<corruption_type>/<severity>/<class>/images

    5. ImageNet-P:
       Download from https://zenodo.org/record/3565846
       Extract to <root_dir>/imagenet-p/
       Contains perturbation sequences as folders of frames.

    6. ImageNet-Sketch:
       Download from https://drive.google.com/open?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
       or from Hugging Face: https://huggingface.co/datasets/imagenet_sketch
       Extract to <root_dir>/imagenet-sketch/
       Contains 50,000 sketch images (50 per class, 1000 classes).

    7. ImageNet-V2 (MatchedFrequency):
       Download from https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main
       Extract to <root_dir>/imagenetv2-matched-frequency-format-val/
       Contains 10,000 images (10 per class, 1000 classes).

    Directory structure:
        <data_path>/imagenet/
        ├── imagenet1k/
        │   ├── train/            (1000 class folders)
        │   └── val/              (1000 class folders)
        ├── imagenet-a/           (200 class folders)
        ├── imagenet-r/           (200 class folders)
        ├── imagenet-c/
        │   ├── gaussian_noise/
        │   │   ├── 1/ ... 5/    (severity levels, each with class folders)
        │   ├── shot_noise/
        │   └── ...
        ├── imagenet-p/
        │   ├── gaussian_noise/
        │   └── ...
        ├── imagenet-sketch/      (1000 class folders)
        └── imagenetv2-matched-frequency-format-val/  (1000 numbered folders)

@references:
- ImageNet: Olga Russakovsky, et al. "ImageNet Large Scale Visual Recognition
    Challenge." IJCV 2015.
- ImageNet-A: Dan Hendrycks, et al. "Natural Adversarial Examples." CVPR 2021.
- ImageNet-R: Dan Hendrycks, et al. "The Many Faces of Robustness." NeurIPS 2021.
- ImageNet-C/P: Dan Hendrycks and Thomas Dietterich. "Benchmarking Neural Network
    Robustness to Common Corruptions and Perturbations." ICLR 2019.
- ImageNet-Sketch: Haohan Wang, et al. "Learning Robust Global Representations
    by Penalizing Local Predictive Power." NeurIPS 2019.
- ImageNet-V2: Benjamin Recht, et al. "Do ImageNet Classifiers Generalize to
    ImageNet?" ICML 2019.
"""

import os
import random
from torch.utils.data import Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
import json
from tqdm import tqdm 
IMAGENET_C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

IMAGENET_C_SEVERITIES = [1, 2, 3, 4, 5]


class TrackedImageFolder(ImageFolder):
    @staticmethod
    def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
        instances = []
        directory = os.path.expanduser(directory)
        
        # We wrap the classes in tqdm to see it crawl through n01443537, n01629819, etc.
        sorted_classes = sorted(class_to_idx.items())
        for target_class, target_idx in tqdm(sorted_classes, desc="Scanning ImageNet Classes"):
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            
            # This part mimics the internal torchvision logic
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    instances.append((path, target_idx))
        return instances
    
class ImageNet(VisionDataset):
    """
    Dataset class for ImageNet-1k and its distribution-shifted test variants.

    Splits:
        - "train": ImageNet-1k training set.
        - "val": ImageNet-1k validation set.
        - "test": ImageNet-1k validation set (in-distribution test).
        - "test_a": ImageNet-A (natural adversarial examples).
        - "test_r": ImageNet-R (renditions).
        - "test_c": ImageNet-C (corruptions, specify via kwargs).
        - "test_p": ImageNet-P (perturbations).
        - "test_sketch": ImageNet-Sketch.
        - "test_v2": ImageNet-V2 (MatchedFrequency).
        - "test_r_c26": ImageNet-R Ablation set.
    """

    VALID_SPLITS = [
        "train", "val", "test", "test_a", "test_r",
        "test_c", "test_p", "test_sketch", "test_v2", "test_abl", "test_r_c26"
    ]

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        """
        Initialize the ImageNet dataset.

        Args:
            root_dir (str): Root directory containing all ImageNet variants.
            split (str): Dataset split (see class docstring).
            transform (callable, optional): Transforms for images.
            target_transform (callable, optional): Transforms for targets.
            use_subset (bool): Whether to use a subset.
            subset_size (int, optional): Subset size.
            subset_seed (int, optional): Subset random seed.
            **kwargs:
                corruption_type (str): Corruption name for test_c (default: all).
                severity (int): Severity level 1-5 for test_c (default: all).
                v2_variant (str): ImageNet-V2 variant name. Defaults to
                    "imagenetv2-matched-frequency-format-val".
        """
        super(ImageNet, self).__init__(root_dir, transform=transform,
                                       target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        if "@" in split:
            a_split, b_split = split.split("@")
        else:
            b_split = split 
            a_split = split
        assert b_split in self.VALID_SPLITS and a_split in self.VALID_SPLITS, \
            f"Split must be one of {self.VALID_SPLITS}, got '{split}'."

        self.dataset = self._load_split(root_dir, split, **kwargs)

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)
    
    def _get_imagenet_linkfolder(self, root_dir:str, split:str, sub:str):
        """ 
        Creates symlink dictionaries for the Imgenet style pool.
        Filters for relevant Classes of the Subset.

        """
        subset_name = f"{split}_{sub}" if split != "train" else sub
        subset_dir = os.path.abspath(os.path.join(root_dir, "imagenet1k/subsets", subset_name))
        if os.path.exists(subset_dir):
            return subset_dir
        os.makedirs(subset_dir, exist_ok=True)

        with open("./data/imagenet/imagenet_subsets.json", "r") as f:
            data = json.load(f)

        subset_classes = data.get(sub, [])
        src_name = split
        src_base = os.path.abspath(os.path.join(root_dir, f'imagenet1k/ILSVRC/Data/CLS-LOC/{src_name}'))
        
        for c in subset_classes:
            link_target = os.path.join(src_base, c)
            link_name = os.path.join(subset_dir,c)
            assert os.path.exists(link_target), f"{link_target} does not lead to a Imagenet Folder"
            if not os.path.exists(link_name):
                os.symlink(link_target, link_name, target_is_directory=True)
        
        return subset_dir

    def _load_split(self, root_dir: str, split: str, **kwargs):
        
        sub = None
        if "@" in split:
            split, sub = split.split("@")

        if split in ["train", "val"]:
            if sub is not None:
                path=self._get_imagenet_linkfolder(root_dir, split, sub)
            else:
                path = os.path.join(root_dir, 'imagenet1k/ILSVRC/Data/CLS-LOC', split)
            self._check_dir(path, f"ImageNet-1k {split}")
            return ImageFolder(path)

        elif split == "test":
            path = os.path.join(root_dir, 'imagenet1k/ILSVRC/Data/CLS-LOC', split)
            self._check_dir(path, f"ImageNet-1k {split}")
            return ImageFolder(path)
        
        elif split == "test_abl":
            path = os.path.join(root_dir,'imagenetabl')
            self._check_dir(path, 'Ablation R DIR')
            return ImageFolder(path)
        
        elif split == "test_a":
            path = os.path.join(root_dir,'imageneta', 'imagenet-a')
            self._check_dir(path, "ImageNet-A")
            return ImageFolder(path)

        elif split == "test_r":
            path = os.path.join(root_dir, 'imagenetr','imagenet-r')
            self._check_dir(path, "ImageNet-R")
            return ImageFolder(path)
        
        elif split == "test_r_c26":
            path = os.path.join(root_dir, 'imagenetr','imagenet-r-c26')
            self._check_dir(path, "ImageNet-R")
            return ImageFolder(path)

        elif split == "test_c":
            return self._load_imagenet_c(root_dir, **kwargs)

        elif split == "test_p":
            path = os.path.join(root_dir, 'imagenet-p')
            self._check_dir(path, "ImageNet-P")
            return ImageFolder(path)

        elif split == "test_sketch":
            path = os.path.join(root_dir, 'imagenet-sketch')
            self._check_dir(path, "ImageNet-Sketch")
            return ImageFolder(path)

        elif split == "test_v2":
            variant = kwargs.get('v2_variant',
                                 'imagenetv2-matched-frequency-format-val')
            path = os.path.join(root_dir, variant)
            self._check_dir(path, "ImageNet-V2")
            return ImageFolder(path)

    def _load_imagenet_c(self, root_dir: str, **kwargs):
        corruption_type = kwargs.get('corruption_type', None)
        severity = kwargs.get('severity', None)

        c_dir = os.path.join(root_dir, 'imagenet-c')
        self._check_dir(c_dir, "ImageNet-C")

        corruptions = [corruption_type] if corruption_type else IMAGENET_C_CORRUPTIONS
        severities = [severity] if severity else IMAGENET_C_SEVERITIES

        datasets = []
        for corr in corruptions:
            for sev in severities:
                path = os.path.join(c_dir, corr, str(sev))
                if os.path.isdir(path):
                    datasets.append(ImageFolder(path))

        if len(datasets) == 0:
            raise FileNotFoundError(
                f"No ImageNet-C data found in {c_dir}. Check directory structure.")
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    @staticmethod
    def _check_dir(path: str, name: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{name} directory not found at {path}. "
                f"Please download and extract the dataset there."
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
