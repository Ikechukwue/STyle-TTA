"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for loading images from a folder structure.
"""

import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple, List
import random


class ImageFolder(VisionDataset):
    """
    A generic dataset class for loading images from a folder.

    The folder structure should be:
        root/
            image1.jpg
            image2.png
            ...

    Supports common image formats: .jpg, .jpeg, .png, .bmp, .tiff
    """

    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_subset: bool = False,
        subset_size: Optional[int] = None,
        subset_seed: Optional[int] = None
    ) -> None:
        """
        Initialize the ImageFolder dataset.

        Args:
            root: Root directory containing images
            transform: Transformations to apply to the images
            target_transform: Transformations to apply to the target
                (unused, for compatibility)
            use_subset: Whether to use only a subset of the dataset
            subset_size: Number of samples to use if use_subset is True
            subset_seed: Random seed for subset selection
        """
        super(ImageFolder, self).__init__(root, transform=transform,
                                          target_transform=target_transform)

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(self.root):
            raise RuntimeError(f'Dataset directory not found: {self.root}')

        # Find all image files in the directory
        self.image_files = self._find_images()

        if len(self.image_files) == 0:
            raise RuntimeError(f'No images found in directory: {self.root}')

        # Sort for reproducibility
        self.image_files.sort()

        # Apply subsetting if requested
        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

        print(f"Loaded {len(self.image_files)} images from {self.root}")

    def _find_images(self) -> List[str]:
        """
        Find all image files in the root directory.

        Returns:
            List of image file paths
        """
        image_files = []

        for filename in os.listdir(self.root):
            if filename.lower().endswith(self.IMAGE_EXTENSIONS):
                image_path = os.path.join(self.root, filename)
                if os.path.isfile(image_path):
                    image_files.append(image_path)

        return image_files

    def _apply_subset(self, subset_size: int,
                      subset_seed: Optional[int] = None) -> None:
        """
        Apply subsetting to the dataset.

        Args:
            subset_size: Number of samples to use for the subset
            subset_seed: Random seed for subset selection.
                If None, uses 42 for reproducibility.
        """
        dataset_size = len(self.image_files)

        if subset_size >= dataset_size:
            print(f"Warning: Requested subset size ({subset_size}) is >= dataset size "
                  f"({dataset_size}). Using entire dataset.")
            return

        # Randomly sample subset_size images
        seed = subset_seed if subset_seed is not None else 42
        random.seed(seed)
        self.image_files = random.sample(self.image_files, subset_size)
        print(f"Using subset of {subset_size} samples from {dataset_size} "
              f"total samples (seed={seed}).")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, dict]:
        """
        Get an image and its metadata.

        Args:
            idx: Index of the image

        Returns:
            Tuple of (image, metadata dict)
        """
        img_path = self.image_files[idx]

        try:
            img = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Warning: Could not load image {img_path}: {e}. Skipping.")
            # Return the next valid image
            return self.__getitem__((idx + 1) % len(self))

        # Extract metadata
        filename = os.path.basename(img_path)
        image_id = os.path.splitext(filename)[0]

        metadata = {
            'image_id': image_id,
            'filename': filename,
            'path': img_path
        }

        if self.transform is not None:
            img = self.transform(img)

        return img, metadata
