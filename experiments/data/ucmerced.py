import os
import random
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple

def tif_loader(path: str) -> Image.Image:
    """Explicitly opens TIF images and converts them to RGB format."""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class UCMerced(VisionDataset):
    """
    Dataset class for the UC Merced Land Use dataset.
    """

    def __init__(self, root_dir: str,
                 split: str, 
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 **kwargs) -> None:
        super(UCMerced, self).__init__(root_dir, transform=transform,
                                       target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        
        subset_name = split
        base_path = os.path.join(root_dir, 'satelite/')
        
        # Check standard extraction directory names for UC Merced
        if os.path.isdir(os.path.join(base_path, 'UCMerced_LandUse', 'Images')):
            source_path = os.path.join(base_path, 'UCMerced_LandUse', 'Images')
        elif os.path.isdir(os.path.join(base_path, 'Images')):
            source_path = os.path.join(base_path, 'Images')
        else:
            source_path = base_path
            
        self._check_dir(source_path, "UC Merced")

        # Determine target folder path
        if subset_name == "mapped":
            path = self._get_linkfolder(root_dir, source_path, subset_name)
        else:
            path = source_path
            
        self.dataset = ImageFolder(path, loader=tif_loader, is_valid_file=lambda x: x.lower().endswith('.tif'))
        
        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _get_linkfolder(self, root_dir: str, source_base: str, subset_name: str) -> str:
        """
        Creates symlink folders for the UCMerced dataset mapping 
        specific classes to target class folders.
        """
        # Exclude 'buildings' to align clean 4-class shared taxonomy
        UCMERCED_MAP = {
            "forest": "forest",
            "denseresidential": "residential",
            "mediumresidential": "residential",
            "sparseresidential": "residential",
            "river": "river",
            "agricultural": "agricultural"
        }

        subset_dir = os.path.abspath(os.path.join(root_dir, "satelite/UCMerced_LandUse/subsets", subset_name))
        
        if os.path.exists(subset_dir):
            return subset_dir
        os.makedirs(subset_dir, exist_ok=True)

        for src_class, target_class in UCMERCED_MAP.items():
            src_path = os.path.join(source_base, src_class)
            target_path = os.path.join(subset_dir, target_class)

            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)

            if os.path.exists(src_path):
                for img_name in os.listdir(src_path):
                    if img_name.lower().endswith('.tif'):
                        src_file = os.path.abspath(os.path.join(src_path, img_name))
                        dst_file = os.path.join(target_path, img_name)
                        if not os.path.exists(dst_file):
                            os.symlink(src_file, dst_file)
            else:
                print(f"Warning: Source class folder {src_path} not found.")

        return subset_dir
    
    @staticmethod
    def _check_dir(path: str, name: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{name} directory not found at {path}. "
                f"Please download and extract the dataset there."
            )

    def _apply_subset(self, subset_size: int, subset_seed: Optional[int] = None) -> None:
        from torch.utils.data import Subset
        dataset_size = len(self.dataset)
        seed = subset_seed if subset_seed is not None else 42
        rng = random.Random(seed)

        if subset_size >= dataset_size:
            print(f"Warning: Requested subset size ({subset_size}) >= dataset size ({dataset_size}).")
            return

        indices = rng.sample(range(dataset_size), subset_size)
        self.dataset = Subset(self.dataset, indices)
        print(f"Using subset of {subset_size} samples (seed={seed}).")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
