import os
import random
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets.vision import VisionDataset
from wilds import get_dataset


class Camelyon17WILDS(VisionDataset):
    """
    Dataset class for the Camelyon17-WILDS dataset.

    By default, behaves exactly like the original implementation.

    Optionally, inject_stylized_images_inplace() can configure this
    dataset to load replacement images from a directory.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_subset: bool = False,
        subset_size: Optional[int] = None,
        subset_seed: Optional[int] = None,
    ) -> None:

        super().__init__(
            root_dir,
            transform=transform,
            target_transform=target_transform,
        )

        self.transform = transform
        self.target_transform = target_transform

        full_dataset = get_dataset(
            dataset="camelyon17",
            download=True,
            root_dir=root_dir,
        )

        self.dataset = full_dataset.get_subset(split)

        # ---------------------------------------------------------
        # Stylized image injection configuration
        # ---------------------------------------------------------
        self._stylized_dir = None
        self._stylized_view_name = None

        # Enforce 2,000 subset restriction specifically for test.
        if split == "test":
            use_subset = True
            subset_size = 2000 if subset_size is None else subset_size

        if use_subset and subset_size is not None:
            self._apply_subset(subset_size, subset_seed)

    def _apply_subset(
        self,
        subset_size: int,
        subset_seed: Optional[int] = None,
    ) -> None:

        dataset_size = len(self.dataset)
        seed = subset_seed if subset_seed is not None else 42

        rng = random.Random(seed)

        if subset_size >= dataset_size:

            repeats = (
                subset_size + dataset_size - 1
            ) // dataset_size

            print(
                f"Warning: Requested subset size ({subset_size}) "
                f"is >= dataset size ({dataset_size}). "
                f"Repeating dataset {repeats}x to fulfill requirements."
            )

            all_indices = []

            for _ in range(repeats):
                indices = list(range(dataset_size))
                rng.shuffle(indices)
                all_indices.extend(indices)

            indices = all_indices[:subset_size]

            self.dataset = Subset(
                self.dataset,
                indices,
            )

            print(
                f"Using subset of {subset_size} samples "
                f"(repeated from {dataset_size} originals, "
                f"seed={seed})."
            )

            return

        indices = rng.sample(
            range(dataset_size),
            subset_size,
        )

        self.dataset = Subset(
            self.dataset,
            indices,
        )

        print(
            f"Using subset of {subset_size} samples from "
            f"{dataset_size} total samples (seed={seed})."
        )

    # =============================================================
    # STYLIZED IMAGE INJECTION
    # =============================================================

    def inject_stylized_images(
        self,
        new_base_dir_path: str,
        view_name: str = "view_001.png",
    ) -> None:
        """
        Configure this dataset to load stylized images instead of
        the original WILDS images.
        """
        self._stylized_dir = new_base_dir_path
        self._stylized_view_name = view_name

        if not os.path.isdir(new_base_dir_path):
            raise FileNotFoundError(
                f"Stylized image directory does not exist: {new_base_dir_path}"
            )

        # 1. Test for Global WILDS Indexing (e.g., 83810)
        first_global_idx = self._get_original_index(0)
        global_path = os.path.join(
            new_base_dir_path,
            f"{first_global_idx:05d}",
            view_name,
        )

        # 2. Test for Sequential Subset Indexing (e.g., 00000)
        sequential_path = os.path.join(
            new_base_dir_path,
            "00000",
            view_name,
        )

        if os.path.exists(global_path):
            self._use_global_index = True
            last_global_idx = self._get_original_index(len(self.dataset) - 1)
            last_path = os.path.join(
                new_base_dir_path, f"{last_global_idx:05d}", view_name
            )
            if not os.path.exists(last_path):
                raise FileNotFoundError(f"Missing stylized image: {last_path}")

        elif os.path.exists(sequential_path):
            self._use_global_index = False
            last_seq_idx = len(self.dataset) - 1
            last_path = os.path.join(
                new_base_dir_path, f"{last_seq_idx:05d}", view_name
            )
            if not os.path.exists(last_path):
                raise FileNotFoundError(f"Missing stylized image: {last_path}")

        else:
            raise FileNotFoundError(
                f"Could not find valid stylized image at either:\n"
                f" - Global path: {global_path}\n"
                f" - Sequential path: {sequential_path}"
            )

        print(f"Injected stylized images from: {new_base_dir_path}")
        print(f"View: {view_name} (Global indexing: {self._use_global_index})")

    def _get_original_index(self, idx: int) -> int:
        """
        Resolve an index through any Subset layers.
        Returns the index belonging to the original WILDS dataset.
        """
        dataset = self.dataset
        current_idx = idx

        while isinstance(dataset, Subset):
            current_idx = dataset.indices[current_idx]
            dataset = dataset.dataset

        return current_idx

    def _get_base_dataset(self):
        """
        Return the underlying Camelyon17 WILDS dataset.
        """
        dataset = self.dataset

        while isinstance(dataset, Subset):
            dataset = dataset.dataset

        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[Any, Any]:

        # Standard behavior
        if self._stylized_dir is None:
            img, target, _ = self.dataset[idx]

        # Stylized behavior
        else:
            if getattr(self, "_use_global_index", False):
                target_idx = self._get_original_index(idx)
                base_dataset = self._get_base_dataset()
                _, target, _ = base_dataset[target_idx]
            else:
                target_idx = idx
                _, target, _ = self.dataset[idx]

            stylized_path = os.path.join(
                self._stylized_dir,
                f"{target_idx:05d}",
                self._stylized_view_name,
            )

            if not os.path.exists(stylized_path):
                raise FileNotFoundError(
                    f"Missing stylized image: {stylized_path}"
                )

            img = Image.open(stylized_path).convert("RGB")

        # Transforms
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
