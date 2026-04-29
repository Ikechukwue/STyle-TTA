"""
ReferenceDatabase — Lazy-loading database for training reference images.
=========================================================================

Instead of loading all training images into a single ``(N, 3, H, W)``
tensor (which can exhaust memory on large datasets), this class wraps a
PyTorch ``Dataset`` and loads images on-the-fly by index.

It extracts **only the labels** in a single pass at init time so that
class-balanced retrieval strategies can work without materialising all
pixel data.  Pre-computed DINO embeddings can also be attached for
embedding-based retrieval.

Usage
-----
>>> from experiments.data import create_dataset
>>> ds = create_dataset("pathmnist", "/data", split="train", transform=tfm)
>>> db = ReferenceDatabase.from_dataset(ds)
>>> img = db.get_image(42)           # (3, H, W) loaded on demand
>>> db.labels                        # (N,) tensor
>>> db.get_indices_for_class(3)      # [12, 45, 78, ...]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class ReferenceDatabase:
    """Lazy reference-image store backed by a PyTorch ``Dataset``.

    Parameters
    ----------
    dataset : Dataset
        A PyTorch dataset where ``dataset[i]`` returns ``(image, label)``.
        Images should be ``(3, H, W)`` in ``[0, 1]``.
    labels : Tensor | None
        Pre-computed ``(N,)`` integer label tensor.  If ``None``, labels
        are extracted in a single pass over the dataset at init time.
    dino_embeddings : Tensor | None
        Optional ``(N, D)`` normalised DINOv2 embeddings.
    max_refs : int | None
        If set, cap the database to this many images (random subset).
        The subset is deterministic for a given *seed*.
    seed : int
        Random seed for reproducible sub-sampling.
    """

    def __init__(
        self,
        dataset: Dataset,
        labels: Optional[torch.Tensor] = None,
        dino_embeddings: Optional[torch.Tensor] = None,
        max_refs: Optional[int] = None,
        seed: int = 0,
    ):
        self.dataset = dataset
        self._dino_embeddings = dino_embeddings

        # ---- sub-sample if requested ----------------------------------------
        full_n = len(dataset)  # type: ignore[arg-type]
        if max_refs is not None and max_refs < full_n:
            g = torch.Generator().manual_seed(seed)
            self._indices = torch.randperm(full_n, generator=g)[:max_refs].sort().values.tolist()
        else:
            self._indices = list(range(full_n))

        self._n = len(self._indices)
        
        # ---- extract labels (one pass, no pixels kept) ----------------------
        if labels is not None:
            if max_refs is not None and max_refs < full_n:
                self._labels = labels[self._indices]
            else:
                self._labels = labels
        else:
            self._labels = self._extract_labels()

        # ---- pre-build class index ------------------------------------------
        self._class_indices: Dict[int, List[int]] = {}
        for local_idx, lbl in enumerate(self._labels.tolist()):
            self._class_indices.setdefault(int(lbl), []).append(local_idx)
        self._classes = sorted(self._class_indices.keys())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        max_refs: Optional[int] = None,
        seed: int = 0,
        dino_embeddings: Optional[torch.Tensor] = None,
    ) -> "ReferenceDatabase":
        """Convenience constructor from a plain PyTorch Dataset."""
        return cls(
            dataset=dataset,
            max_refs=max_refs,
            seed=seed,
            dino_embeddings=dino_embeddings,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n

    def get_image(self, local_idx: int) -> torch.Tensor:
        """Load a single image by *local* index.  Returns ``(3, H, W)``."""
        real_idx = self._indices[local_idx]
        img, _ = self.dataset[real_idx]
        if isinstance(img, torch.Tensor):
            return img
        # Fallback for PIL/numpy datasets
        from torchvision.transforms import v2
        return v2.functional.to_image(img).float() / 255.0

    @property
    def labels(self) -> torch.Tensor:
        """``(N,)`` integer labels."""
        return self._labels

    @property
    def classes(self) -> List[int]:
        return self._classes

    def get_indices_for_class(self, cls: int) -> List[int]:
        """Return local indices belonging to *cls*."""
        return self._class_indices.get(cls, [])

    @property
    def dino_embeddings(self) -> Optional[torch.Tensor]:
        return self._dino_embeddings

    @dino_embeddings.setter
    def dino_embeddings(self, value: torch.Tensor) -> None:
        self._dino_embeddings = value

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_labels(self) -> torch.Tensor:
        """Iterate the dataset once to collect labels (no images stored)."""
        def debug_dataset(ds, level=0):
            print(f"[Level {level}] Type: {type(ds)}")
            print(f"    - Attributes: {[a for a in dir(ds) if not a.startswith('__')]}")
            if hasattr(ds, 'dataset'):
                print(f"    - Found '.dataset' wrapper, digging deeper...")
                debug_dataset(ds.dataset, level + 1)
            elif hasattr(ds, 'datasets'):
                print(f"    - Found '.datasets' (ConcatDataset), checking first element...")
                debug_dataset(ds.datasets[0], level + 1)
        curr = self.dataset
        while curr is not None:
                # Check for common label attributes
                for attr in ["targets", "labels", "samples"]:
                    if hasattr(curr, attr):
                        data = getattr(curr, attr)
                        
                        # ImageFolder stores 'samples' as [(path, class), ...]
                        if attr == "samples":
                            raw_labels = [s[1] for s in data]
                        else:
                            raw_labels = data
                        
                        # Convert to tensor and apply the local indexing/subsampling
                        if not isinstance(raw_labels, torch.Tensor):
                            raw_labels = torch.tensor(raw_labels, dtype=torch.long)
                        
                        return raw_labels[self._indices]
                
                # Move to the next nested layer if it exists
                curr = getattr(curr, "dataset", None)

            # Extreme Fallback: only if the above search fails
        print("Direct metadata access failed. Falling back to slow iteration...")
        lbls: List[int] = []
        for local_idx in tqdm(range(self._n)):
            real_idx = self._indices[local_idx]
            _, y = self.dataset[real_idx]
            if isinstance(y, torch.Tensor):
                lbls.append(int(y.item()) if y.numel() == 1 else int(y[0].item()))
            else:
                lbls.append(int(y))
        return torch.tensor(lbls, dtype=torch.long)
