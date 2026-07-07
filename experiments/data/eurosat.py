import os
import random
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple

class EuroSAT(VisionDataset):
    """
    Dataset class for EuroSAT with support for class mapping via '@' syntax.
    
    Example: 
        EuroSAT(root_dir='...', split='train@mapped')
    """
    
    VALID_SPLITS = ["train", "val", "test", "ucmerced"]
    
    EUROSAT_MAP = {
        "Forest": "forest",
        "Residential": "residential",
        "River": "river",
        "Industrial": "industrial",
        "AnnualCrop": "agricultural",
        "PermanentCrop": "agricultural",
        "Pasture": "agricultural"
    }

    def __init__(self, root_dir: str, split: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 **kwargs) -> None:
        super(EuroSAT, self).__init__(root_dir, transform=transform,
                                      target_transform=target_transform)
        
        # Handle split@mapping syntax
        mapping_key = None
        if "@" in split:
            main_split, mapping_key = split.split("@")
        else:
            main_split = split
            
        assert main_split in self.VALID_SPLITS, f"Split must be one of {self.VALID_SPLITS}"
        
        # Determine base path
        base_path = os.path.join(root_dir, 'satelite')
        if os.path.isdir(os.path.join(base_path, '21_classes')):
            source_path = os.path.join(base_path, '21_classes')
        elif os.path.isdir(os.path.join(base_path, 'EuroSAT')):
            source_path = os.path.join(base_path, 'EuroSAT')
        else:
            source_path = base_path

        # Determine if we need to link
        if mapping_key == "ucmerced": # legacy path
            path = self._get_linkfolder(root_dir, source_path)
        else:
            path = source_path

        self._check_dir(path, "EuroSAT")
        
        
        self.dataset = self._get_split(path, main_split, train_ratio, val_ratio)
        
        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _get_linkfolder(self, root_dir: str, source_base: str) -> str:
        subset_dir = os.path.abspath(os.path.join(root_dir, "satelite/EuroSAT_mapped"))
        if os.path.exists(subset_dir):
            return subset_dir
        
        os.makedirs(subset_dir, exist_ok=True)
        for src_class, target_class in self.EUROSAT_MAP.items():
            src_path = os.path.abspath(os.path.join(source_base, src_class))
            target_path = os.path.join(subset_dir, target_class)
            os.makedirs(target_path, exist_ok=True)
            
            if os.path.exists(src_path):
                for img_name in os.listdir(src_path):
                    if not os.path.exists(os.path.join(target_path, img_name)):
                        os.symlink(os.path.join(src_path, img_name), 
                                   os.path.join(target_path, img_name))
        return subset_dir

    def _get_split(self, path: str, split: str, train_ratio: float, val_ratio: float) -> Subset:
        
        if split == "ucmerced":
            data_path = "/home/stud/nemmler/retristyle/data/satelite/UCMerced_LandUse/subsets/mapped"
            dataset = ImageFolder(data_path)
            split_indices = list(range(len(dataset)))
            print("Loaded UCMerced subset")
        else:   
            dataset = ImageFolder(path)
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            rng = random.Random(42)
            rng.shuffle(indices)
            
            train_end = int(train_ratio * dataset_size)
            val_end = train_end + int(val_ratio * dataset_size)

            if split == "train":
                split_indices = indices[:train_end]
            elif split == "val":
                split_indices = indices[train_end:val_end]
            else:
                split_indices = indices[val_end:]
        return Subset(dataset, split_indices)

    @staticmethod
    def _check_dir(path: str, name: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{name} directory not found at {path}.")

    def _apply_subset(self, subset_size: int, subset_seed: Optional[int] = None) -> None:
        dataset_size = len(self.dataset)
        seed = subset_seed if subset_seed is not None else 42
        rng = random.Random(seed)
        if subset_size < dataset_size:
            indices = rng.sample(range(dataset_size), subset_size)
            self.dataset = Subset(self.dataset, indices)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target
