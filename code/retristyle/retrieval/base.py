"""
Base interface for all retrieval methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .reference_db import ReferenceDatabase


class BaseRetriever(ABC):
    """Abstract base for every retriever.

    Sub-classes must implement :meth:`retrieve` which, given a test image,
    returns a set of training-set indices (and optionally scores).

    A ``ReferenceDatabase`` is used as the backing store so that images
    are loaded lazily (on demand) rather than all held in memory.
    """

    @abstractmethod
    def retrieve(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[List[int], Optional[List[float]]]:
        """Return *k* training-set indices (and optional similarity scores).

        Parameters
        ----------
        query : Tensor
            Test image ``(1, 3, H, W)`` in [0, 1].
        k : int
            Number of references to retrieve.

        Returns
        -------
        indices : list[int]
            Indices into the training set.
        scores : list[float] | None
            Similarity scores (higher = better); ``None`` if N/A.
        """
        ...

    def get_image(self, idx: int) -> torch.Tensor:
        """Load a single reference image by index.  Returns ``(3, H, W)``."""
        if hasattr(self, "db") and self.db is not None:
            return self.db.get_image(idx)
        # Legacy fallback: in-memory tensor
        if hasattr(self, "images") and self.images is not None:
            return self.images[idx]
        raise RuntimeError("No image source available (set .db or .images)")

    def get_images(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, List[int], Optional[List[float]]]:
        """Retrieve indices then return the actual image tensors."""
        indices, scores = self.retrieve(query, k=k)
        imgs = torch.stack([self.get_image(i) for i in indices])
        return imgs, indices, scores
