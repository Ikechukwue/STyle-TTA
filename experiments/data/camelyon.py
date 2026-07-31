import random
from torch.utils.data import Subset
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from wilds import get_dataset


class Camelyon17WILDS(VisionDataset):
    """
    Dataset class for the Camelyon17-WILDS dataset.
    """

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, use_subset: bool = False,
                 subset_size: Optional[int] = None,
                 subset_seed: Optional[int] = None) -> None:
        super(Camelyon17WILDS, self).__init__(root_dir, transform=transform,
                                              target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform

        full_dataset = get_dataset(dataset="camelyon17", download=True,
                                   root_dir=root_dir)
        self.dataset = full_dataset.get_subset(split)

        # Enforce 2,000 subset restriction specifically for test split
        if split == "test":
            use_subset = True
            subset_size = 2000 if subset_size is None else subset_size

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _apply_subset(self, subset_size: int,
                      subset_seed: Optional[int] = None) -> None:
        dataset_size = len(self.dataset)
        seed = subset_seed if subset_seed is not None else 42
        random.seed(seed)

        if subset_size >= dataset_size:
            repeats = (subset_size + dataset_size - 1) // dataset_size
            print(f"Warning: Requested subset size ({subset_size}) is >= dataset size "
                  f"({dataset_size}). Repeating dataset {repeats}x to fulfill "
                  f"requirements.")

            all_indices = []
            for _ in range(repeats):
                indices = list(range(dataset_size))
                random.shuffle(indices)
                all_indices.extend(indices)

            indices = all_indices[:subset_size]
            self.dataset = Subset(self.dataset, indices)
            print(f"Using subset of {subset_size} samples (repeated from "
                  f"{dataset_size} originals, seed={subset_seed}).")
            return

        indices = random.sample(range(dataset_size), subset_size)
        self.dataset = Subset(self.dataset, indices)

        print(f"Using subset of {subset_size} samples from "
              f"{dataset_size} total samples (seed={subset_seed}).")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target, _ = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
