import os
import random
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple, List, Dict
from tqdm import tqdm
from config.helpers import load_json


class Midog2022(VisionDataset):
    """
    Dataset class for the MIDOG 2022 dataset that extracts image patches 
    centered around annotated bounding boxes with custom domain-generalization splits.
    Pre-crops patches into 'midog22_dataset/patch_images/<split_name>/<label>' 
    and wraps an internal ImageFolder dataset for efficient loading.
    """
    VALID_SPLITS = ["train", "val", "test", "breasts", "train@breasts", "val@breasts"]

    SCANNER_A = list(range(1, 151))   
    SCANNER_B = list(range(151, 195)) 
    SCANNER_C = list(range(195, 250)) 
    SCANNER_D = list(range(250, 300)) 
    SCANNER_E = list(range(300, 355)) 
    SCANNER_F = list(range(355, 406)) 

    def __init__(self, root_dir: str, split: str,
                 patch_size: int = 224,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 train_ratio: float = 0.8,
                 json_path: Optional[str] = None,
                 **kwargs) -> None:
        super(Midog2022, self).__init__(root_dir, transform=transform,
                                       target_transform=target_transform)

        self.patch_size = patch_size

        if "@" in split:
            main_split, mapping_key = split.split("@")
            resolved_split = f"{main_split}@{mapping_key}"
        else:
            resolved_split = split

        assert resolved_split in self.VALID_SPLITS, f"Split must be one of {self.VALID_SPLITS}"
        self.split_name = resolved_split

        self.images_dir = os.path.join(root_dir, "midog22_dataset/images")
        self.patch_dir = os.path.join(root_dir, "midog22_dataset/patch_images", self.split_name)
        self._check_dir(self.images_dir, "MIDOG Images")

        if json_path is None:
            json_path = os.path.join(self.images_dir, "MIDOG2022_training_png.json")
        self.coco_data = load_json(json_path)

        samples = self._get_split(self.split_name, train_ratio)
        self._cache_patches(samples)

        self.dataset = ImageFolder(self.patch_dir)

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _get_split(self, split: str, train_ratio: float) -> List[Dict[str, Any]]:
            if split == "breasts":
                valid_ids = set(range(101, 151))
            elif split == "train@breasts":
                valid_ids = set(range(1, 91))
            elif split == "val@breasts":
                valid_ids = set(range(91, 101))
            elif split in ["train", "val"]:
                train_val_cases = self.SCANNER_A + self.SCANNER_B + self.SCANNER_C + self.SCANNER_E
                rng = random.Random(42)
                rng.shuffle(train_val_cases)
                split_idx = int(train_ratio * len(train_val_cases))
                valid_ids = set(train_val_cases[:split_idx]) if split == "train" else set(train_val_cases[split_idx:])
            elif split == "test":
                valid_ids = set(self.SCANNER_D)

            img_to_anns = {}
            for ann in self.coco_data.get("annotations", []):
                img_id = ann.get("image_id")
                if img_id in valid_ids:
                    img_to_anns.setdefault(img_id, []).append(ann)

            img_dimensions = {img["id"]: (img["width"], img["height"]) for img in self.coco_data.get("images", [])}

            samples = []

            for img_id, anns in img_to_anns.items():
                # Prioritize class 1 (mitosis) over class 0 so positive annotations take precedence on collision
                sorted_anns = sorted(anns, key=lambda a: 0 if a.get("category_id") == 1 else 1)
                
                img_w, img_h = img_dimensions.get(img_id, (7000, 5000))
                seen_crop_boxes = set()

                for ann in sorted_anns:
                    x, y, w, h = ann["bbox"]
                    cx = x + (w / 2.0)
                    cy = y + (h / 2.0)

                    # Compute the exact clamped coordinates as done in _cache_patches
                    left = int(round(cx - self.patch_size / 2.0))
                    top = int(round(cy - self.patch_size / 2.0))
                    left = max(0, min(left, img_w - self.patch_size))
                    top = max(0, min(top, img_h - self.patch_size))
                    crop_box = (left, top, left + self.patch_size, top + self.patch_size)

                    # Skip if this exact pixel crop box was already generated for this image
                    if crop_box in seen_crop_boxes:
                        continue

                    seen_crop_boxes.add(crop_box)
                    ann_id = ann["id"]
                    label = 1 if ann.get("category_id") == 1 else 0
                    patch_path = os.path.join(self.patch_dir, str(label), f"patch_{ann_id}.png")

                    samples.append({
                        "ann_id": ann_id,
                        "image_id": img_id,
                        "cx": cx,
                        "cy": cy,
                        "patch_path": patch_path,
                        "label": label
                    })

            return samples

    def _cache_patches(self, samples: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.join(self.patch_dir, "0"), exist_ok=True)
        os.makedirs(os.path.join(self.patch_dir, "1"), exist_ok=True)

        missing_samples = [s for s in samples if not os.path.exists(s["patch_path"])]
        if not missing_samples:
            return

        img_to_samples = {}
        for sample in missing_samples:
            img_to_samples.setdefault(sample["image_id"], []).append(sample)

        print(f"\n[MIDOG2022] Pre-cropping {len(missing_samples)} patches for split '{self.split_name}' into class subfolders...")
        
        for img_id, samples_list in tqdm(img_to_samples.items(), desc=f"Caching {self.split_name}"):
            img_name = f"{img_id:03d}.png"
            img_path = os.path.join(self.images_dir, img_name)

            if not os.path.exists(img_path):
                continue

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                for sample in samples_list:
                    if os.path.exists(sample["patch_path"]):
                        continue

                    left = int(round(sample["cx"] - self.patch_size / 2.0))
                    top = int(round(sample["cy"] - self.patch_size / 2.0))

                    # Clamp so the box never leaves the image, preserving patch_size x patch_size
                    left = max(0, min(left, img.width - self.patch_size))
                    top = max(0, min(top, img.height - self.patch_size))
                    right = left + self.patch_size
                    bottom = top + self.patch_size

                    patch = img.crop((left, top, right, bottom))

                    patch.save(sample["patch_path"])

    @staticmethod
    def _check_dir(path: str, name: str) -> None:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{name} directory not found at {path}.")

    def _apply_subset(self, subset_size: int, subset_seed: Optional[int] = None) -> None:
        from torch.utils.data import Subset
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
