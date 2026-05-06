"""
MetricRetriever — Structure-metric retrieval using SSIM on gradient maps
or Mutual Information.

Reference: Sections 1.4, 3.4 of the RetriStyle-TTA report.

Computes a structure-aware similarity between the query and every training
image, then returns the top-k most similar.  The metric operates on
gradient (edge) images so that it is largely invariant to colour/stain shifts.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import BaseRetriever


# ======================================================================
# Utility functions
# ======================================================================
def _to_gray(images: torch.Tensor) -> torch.Tensor:
    """Convert ``(B, 3, H, W)`` RGB to ``(B, 1, H, W)`` grayscale."""
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=images.device, dtype=images.dtype)
    return (images * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    """Compute Sobel gradient magnitude ``(B, 1, H, W)``."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx**2 + gy**2).sqrt()


def ssim_on_gradients(
    a: torch.Tensor, b: torch.Tensor, window_size: int = 11
) -> float:
    """Compute SSIM between Sobel gradient maps of *a* and *b*.

    Both inputs: ``(1, 3, H, W)`` in [0, 1].
    """
    ga = _sobel_edges(_to_gray(a))
    gb = _sobel_edges(_to_gray(b))

    C1, C2 = 0.01**2, 0.03**2
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=a.device, dtype=a.dtype) / (window_size**2)

    mu_a = F.conv2d(ga, kernel, padding=pad)
    mu_b = F.conv2d(gb, kernel, padding=pad)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sig_a2 = F.conv2d(ga * ga, kernel, padding=pad) - mu_a2
    sig_b2 = F.conv2d(gb * gb, kernel, padding=pad) - mu_b2
    sig_ab = F.conv2d(ga * gb, kernel, padding=pad) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / (
        (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    )
    return ssim_map.mean().item()


def mutual_information(
    a: torch.Tensor, b: torch.Tensor, bins: int = 64
) -> float:
    """Estimate MI between grayscale gradient maps via histogram binning."""
    ga = _sobel_edges(_to_gray(a)).flatten()
    gb = _sobel_edges(_to_gray(b)).flatten()

    ga = (ga / (ga.max() + 1e-8) * (bins - 1)).long().clamp(0, bins - 1)
    gb = (gb / (gb.max() + 1e-8) * (bins - 1)).long().clamp(0, bins - 1)

    joint = torch.zeros(bins, bins, device=a.device, dtype=torch.float32)
    for x, y in zip(ga.tolist(), gb.tolist()):
        joint[x, y] += 1.0
    joint = joint / joint.sum()

    pa = joint.sum(dim=1)
    pb = joint.sum(dim=0)
    outer = pa.unsqueeze(1) * pb.unsqueeze(0) + 1e-10
    mi = (joint * torch.log(joint / outer + 1e-10)).sum()
    return mi.item()


# ======================================================================
# Retriever
# ======================================================================
class MetricRetriever(BaseRetriever):
    """Retrieve top-k references by structure-aware metric.

    Parameters
    ----------
    db : ReferenceDatabase | None
        Lazy reference-image database.
    images : Tensor | None
        **(deprecated)** In-memory images ``(N, 3, H, W)``.
    labels : Tensor | None
        **(deprecated)** In-memory labels ``(N,)``.
    metric : ``'ssim'`` | ``'mi'``
        Similarity metric to use.
    precomputed_edges : Tensor | None
        Optional pre-computed Sobel edge maps ``(N, 1, H, W)`` to skip
        redundant computation.
    """

    def __init__(
        self,
        db=None,
        *,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        metric: Literal["ssim", "mi"] = "ssim",
        precomputed_edges: Optional[torch.Tensor] = None,
    ):
        if db is not None:
            from .reference_db import ReferenceDatabase
            self.db: Optional[ReferenceDatabase] = db
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

        self.metric = metric

        # Pre-computed edges (only possible when images are in-memory)
        if precomputed_edges is not None:
            self._edges = precomputed_edges
        elif self.images is not None:
            self._edges = _sobel_edges(_to_gray(self.images))
        else:
            self._edges = None  # will compute on-the-fly

    def retrieve(
        self, query: torch.Tensor, k: int = 5
    ) -> Tuple[List[int], Optional[List[float]]]:
        if self.metric == "ssim":
            scores = self._ssim_scores(query)
        else:
            scores = self._mi_scores(query)

        topk = torch.topk(torch.tensor(scores), k=min(k, len(scores)))
        return topk.indices.tolist(), topk.values.tolist()

    # ------------------------------------------------------------------
    def _get_edge(self, i: int) -> torch.Tensor:
        """Return Sobel edge map for reference *i*.  ``(1, 1, H, W)``."""
        if self._edges is not None:
            return self._edges[i : i + 1]
        img = self.get_image(i).unsqueeze(0)
        return _sobel_edges(_to_gray(img))

    def _ssim_scores(self, query: torch.Tensor) -> List[float]:
        """Compute SSIM-on-gradients between query and every training image."""
        q_edge = _sobel_edges(_to_gray(query))
        scores: List[float] = []
        C1, C2 = 0.01**2, 0.03**2
        for i in range(self.n):
            ref_edge = self._get_edge(i).to(q_edge.device)
            # Resize to match if needed
            if ref_edge.shape[2:] != q_edge.shape[2:]:
                ref_edge = F.interpolate(ref_edge, size=q_edge.shape[2:], mode="bilinear", align_corners=False)
            mu_a = q_edge.mean()
            mu_b = ref_edge.mean()
            sig_a2 = q_edge.var()
            sig_b2 = ref_edge.var()
            sig_ab = ((q_edge - mu_a) * (ref_edge - mu_b)).mean()
            ssim_val = (
                (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
            ) / ((mu_a**2 + mu_b**2 + C1) * (sig_a2 + sig_b2 + C2))
            scores.append(ssim_val.item())
        return scores

    def _mi_scores(self, query: torch.Tensor) -> List[float]:
        """Compute MI between query and every training image."""
        scores: List[float] = []
        for i in range(self.n):
            ref = self.get_image(i).unsqueeze(0)
            scores.append(mutual_information(query, ref))
        return scores
