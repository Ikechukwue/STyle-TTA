"""
RandomRetriever — Uniform random sampling from the training set.

Reference: Section 6.4 of the RetriStyle-TTA report.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .base import BaseRetriever
from .reference_db import ReferenceDatabase


class RandomRetriever(BaseRetriever):
    """Select *k* references uniformly at random.

    Parameters
    ----------
    db : ReferenceDatabase
        Lazy reference-image database.
    images : Tensor | None
        **(deprecated)** In-memory images ``(N, 3, H, W)``.  Kept for
        backward compatibility; prefer *db*.
    labels : Tensor | None
        **(deprecated)** In-memory labels.
    """

    def __init__(
        self,
        db: ReferenceDatabase | None = None,
        *,
        images: torch.Tensor | None = None,
        labels: Optional[torch.Tensor] = None,
        seed: int = 0,
    ):
        if db is not None:
            self.db = db
            self.images = None
            self.labels = db.labels
            self.n = len(db)
            
        elif images is not None:
            self.db = None
            self.images = images
            self.labels = labels
            self.n = images.shape[0]
        else:
            raise ValueError("Either db or images must be provided")
        self.generator = torch.Generator().manual_seed(seed)
    def retrieve(
        self, query: torch.Tensor, k: int = 5
    ) -> Tuple[List[int], Optional[List[float]]]:
        indices = torch.randperm(self.n, generator=self._generator)[:k].tolist()
        return indices, None
