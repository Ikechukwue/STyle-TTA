"""
Usese the image assets of the StyleID paper for the style evaluation script
"""

import os
import random
from pathlib import Path
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple

class Assets(VisionDataset):
    def __init__(
        self,
        root_dir: str,          # Points to: data_path / dataset_name
        split: str,             # Folder name: e.g., "set_20" or "set_40"
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_subset: bool = False,
        subset_size: Optional[int] = None,
        subset_seed: Optional[int] = None,
    ):
        super().__init__(root_dir, transform=transform, target_transform=target_transform)
        
        self.folder_path = Path(root_dir) / split
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Image directory not found: {self.folder_path}")

        # Gather all images regardless of folder structure inside
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        all_images = sorted([
            p for p in self.folder_path.rglob("*") 
            if p.suffix.lower() in valid_extensions
        ])
        
        if not all_images:
            raise RuntimeError(f"No valid images found in {self.folder_path}")

        # Reproducible subsetting
        if use_subset and subset_size is not None:
            if subset_size > len(all_images):
                print(f"Warning: Requested subset_size ({subset_size}) exceeds available "
                      f"images ({len(all_images)}) in {split}. Using all available images.")
                subset_size = len(all_images)
            
            # Localized RNG ensures source & reference seeds don't clobber each other
            rng = random.Random(subset_seed if subset_seed is not None else 42)
            self.image_paths = rng.sample(all_images, subset_size)
        else:
            self.image_paths = all_images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        img_path = self.image_paths[index]
        
        with open(img_path, "rb") as f:
            img = Image.convert("RGB") if hasattr(Image, "convert") else Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # Return a dummy label (0) to match standard PyTorch DataLoader expectations
        return img, 0
