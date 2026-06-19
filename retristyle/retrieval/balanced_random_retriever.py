"""
BalancedRandomRetriever — Class-balanced random sampling.

Reference: Section 4.1.3 of the RetriStyle-TTA report.

Ensures that the retrieved references are drawn equally from each class in
the training set, filling from the most-populated class when a class has
fewer candidates than requested.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from .base import BaseRetriever
from .reference_db import ReferenceDatabase


class BalancedRandomRetriever(BaseRetriever):
    """Retrieve *k* references with equal representation per class.

    Parameters
    ----------
    db : ReferenceDatabase
        Lazy reference-image database.
    images : Tensor | None
        **(deprecated)** In-memory images ``(N, 3, H, W)``.
    labels : Tensor | None
        **(deprecated)** In-memory labels ``(N,)``.
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
            self._class_indices = {c: db.get_indices_for_class(c)
                                   for c in db.classes}
            self._classes = db.classes
        elif images is not None:
            self.db = None
            self.images = images
            self.labels = labels
            self.n = images.shape[0]
            self._class_indices: Dict[int, List[int]] = {}
            for i, lbl in enumerate(labels.tolist()):
                self._class_indices.setdefault(int(lbl), []).append(i)
            self._classes = sorted(self._class_indices.keys())
        else:
            raise ValueError("Either db or images must be provided")
        
        self._generator = torch.Generator().manual_seed(seed)
    def retrieve(
        self, query: torch.Tensor, k: int = 5
    ) -> Tuple[List[int], Optional[List[float]]]:
        n_classes = len(self._classes)
        per_class = k // n_classes
        remainder = k % n_classes

        shuffled_class_indices = torch.randperm(n_classes, generator=self._generator).tolist()
        extra_slots = set(shuffled_class_indices[:remainder])

        selected: List[int] = []
        for ci, cls in enumerate(self._classes):
            pool = self._class_indices[cls]
            need = per_class + (1 if ci in extra_slots else 0)

            need = min(need, len(pool))
            
            if need > 0:
                perm = torch.randperm(len(pool), generator=self._generator)[:need]
                selected.extend([pool[j] for j in perm.tolist()])

        if len(selected) < k:
            selected_set = set(selected)
            remaining_pool = []
            
            while len(selected) < k:
                added_any = False
                for cls in self._classes:
                    pool = self._class_indices[cls]
                    available = [idx for idx in pool if idx not in selected_set]
                    if available:
                        rand_idx = torch.randint(0, len(available), (1,), generator=self._generator).item()
                        chosen = available[rand_idx]
                        selected.append(chosen)
                        selected_set.add(chosen)
                        added_any = True
                        if len(selected) == k:
                            break
                if not added_any:
                    break

        return selected[:k], None
