"""
BalancedMetricRetriever — Class-balanced structure retrieval using MMR.

Reference: Section 4.1.3 of the RetriStyle-TTA report.

Uses Maximal Marginal Relevance (MMR) re-ranking: within each class bucket,
selects references that are structurally similar to the query while being
colour-diverse with respect to each other.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseRetriever
from .metric_retriever import (
    MetricRetriever,
    _to_gray,
    _sobel_edges,
    ssim_on_gradients,
)
from .reference_db import ReferenceDatabase


def _mean_lab_vector(image: torch.Tensor) -> torch.Tensor:
    """Cheap colour summary: channel-wise mean of the image ``(3,)``."""
    return image.mean(dim=[0, 2, 3])  # (3,)


class BalancedMetricRetriever(BaseRetriever):
    """Retrieve *k* references across classes with intra-class diversity.

    Algorithm (from Section 4.1.3):
    1. Get a large pool of candidates (top-M by structure metric).
    2. Bucket candidates by class label.
    3. Within each bucket, greedily select samples that maximise structural
       similarity to the query while minimising colour similarity to
       already-selected samples (MMR).
    4. Fill empty buckets from the nearest non-empty bucket.

    Parameters
    ----------
    db : ReferenceDatabase | None
        Lazy reference-image database.
    images : Tensor | None
        **(deprecated)** In-memory images ``(N, 3, H, W)``.
    labels : Tensor | None
        **(deprecated)** In-memory labels ``(N,)``.
    metric : ``'ssim'`` | ``'mi'``
        Structure metric for the base retriever.
    pool_size : int
        Number of initial candidates (M) to short-list before re-ranking.
    lambda_mmr : float
        Trade-off between relevance and diversity in MMR.
    """

    def __init__(
        self,
        db: ReferenceDatabase | None = None,
        *,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        metric: Literal["ssim", "mi"] = "ssim",
        pool_size: int = 50,
        lambda_mmr: float = 0.5,
    ):
        if db is not None:
            self.db: Optional[ReferenceDatabase] = db
            self.images = None
            self.labels = db.labels
            self.n = len(db)
            self._class_indices = {c: db.get_indices_for_class(c) for c in db.classes}
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

        self.metric = metric
        self.pool_size = pool_size
        self.lambda_mmr = lambda_mmr

        # Build the base metric retriever for the initial pool
        if db is not None:
            self._base = MetricRetriever(db=db, metric=metric)
        else:
            self._base = MetricRetriever(images=images, labels=labels, metric=metric)

        # Colour descriptors: computed lazily on first access
        self._color_desc: Optional[torch.Tensor] = None
        if self.images is not None:
            self._color_desc = torch.stack(
                [_mean_lab_vector(images[i : i + 1]) for i in range(images.shape[0])]
            )  # (N, 3)

    def retrieve(
        self, query: torch.Tensor, k: int = 5
    ) -> Tuple[List[int], Optional[List[float]]]:
        # Step 1: large candidate pool
        pool_indices, pool_scores = self._base.retrieve(query, k=self.pool_size)
        if pool_scores is None:
            pool_scores = [0.0] * len(pool_indices)
        score_map = dict(zip(pool_indices, pool_scores))

        # Step 2: bucket by class
        buckets: Dict[int, List[int]] = {c: [] for c in self._classes}
        for idx in pool_indices:
            cls = int(self.labels[idx].item())
            buckets[cls].append(idx)

        # Step 3: distribute quota
        n_classes = len(self._classes)
        per_class = k // n_classes
        remainder = k % n_classes

        selected: List[int] = []
        selected_scores: List[float] = []

        for ci, cls in enumerate(self._classes):
            need = per_class + (1 if ci < remainder else 0)
            bucket = buckets[cls]
            if not bucket:
                continue  # handle in step 4

            chosen = self._mmr_select(bucket, score_map, need)
            selected.extend(chosen)
            selected_scores.extend([score_map.get(c, 0.0) for c in chosen])

        # Step 4: fill shortfall from any remaining pool candidates
        remaining = [i for i in pool_indices if i not in selected]
        while len(selected) < k and remaining:
            nxt = remaining.pop(0)
            selected.append(nxt)
            selected_scores.append(score_map.get(nxt, 0.0))

        return selected[:k], selected_scores[:k]

    # ------------------------------------------------------------------
    def _get_color_desc(self, idx: int) -> torch.Tensor:
        """Return colour descriptor for reference *idx*.  ``(3,)``."""
        if self._color_desc is not None:
            return self._color_desc[idx]
        # Lazy: load image and compute on-the-fly
        img = self.get_image(idx).unsqueeze(0)  # (1, 3, H, W)
        return _mean_lab_vector(img)

    def _mmr_select(
        self,
        candidates: List[int],
        score_map: Dict[int, float],
        need: int,
    ) -> List[int]:
        """Greedy MMR selection within a class bucket."""
        chosen: List[int] = []
        remaining = list(candidates)

        for _ in range(need):
            if not remaining:
                break
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = score_map.get(idx, 0.0)
                if chosen:
                    # Max colour similarity to already-chosen
                    c_desc = self._get_color_desc(idx)
                    max_sim = max(
                        1.0 / (1.0 + (c_desc - self._get_color_desc(c)).norm().item())
                        for c in chosen
                    )
                else:
                    max_sim = 0.0
                mmr = self.lambda_mmr * relevance - (1 - self.lambda_mmr) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            if best_idx >= 0:
                chosen.append(best_idx)
                remaining.remove(best_idx)

        return chosen
