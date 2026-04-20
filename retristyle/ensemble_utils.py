"""
Ensemble utilities — OOD Filtering (FOODS) and Soft Voting.

Reference: Sections 1.4, 4.3.1, 4.3.2 of the RetriStyle-TTA report.

OOD Filtering
-------------
Extract penultimate features of augmented images, compute distance to
training-set class centroids, and discard augmentations past a safety
threshold τ.

Soft Voting
-----------
Weighted aggregation of predictions using inverse distance to the
training manifold, yielding the final argmax classification.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Feature extraction helper
# ======================================================================
def extract_penultimate_features(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Forward *images* through *model* and intercept penultimate features.

    Works with ``timm`` models (``model.forward_features`` +
    ``model.global_pool``).

    Parameters
    ----------
    model : Module
        Frozen classifier.
    images : Tensor
        ``(B, 3, H, W)`` input.

    Returns
    -------
    Tensor
        ``(B, D)`` penultimate features.
    """
    # Try timm-style API first
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        feats = model.forward_features(images)
        # Pool but skip the final FC
        if hasattr(model, "global_pool"):
            if callable(model.global_pool):
                feats = model.global_pool(feats)
            # If global_pool is a string (timm config), do adaptive avg pool
            elif isinstance(feats, torch.Tensor) and feats.ndim == 4:
                feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            elif feats.ndim == 3:
                # ViT outputs (B, tokens, D) — use CLS token
                feats = feats[:, 0]
        elif feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        elif feats.ndim == 3:
            feats = feats[:, 0]
        return feats

    # Fallback — register hook on the layer before the final FC
    features: List[torch.Tensor] = []

    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            features.append(out.detach())
        else:
            features.append(inp[0].detach())

    # Typically the last child before the classifier head
    children = list(model.children())
    handle = children[-1].register_forward_hook(_hook) if children else None
    model(images)
    if handle:
        handle.remove()
    if features:
        f = features[0]
        if f.ndim == 4:
            f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return f
    raise RuntimeError("Could not extract penultimate features.")


# ======================================================================
# FOODS — OOD Filtering
# ======================================================================
class FOODSFilter:
    """Filtering OOD Samples via distance to training-set class centroids.

    Parameters
    ----------
    model : Module
        Frozen classifier.
    train_images : Tensor
        ``(N, 3, H, W)`` training images in [0, 1].
    train_labels : Tensor
        ``(N,)`` labels.
    tau : float
        Safety threshold.  Augmented views with distance > τ are discarded.
    batch_size : int
        Batch size for centroid computation.
    normalize_fn : callable | None
        Optional normalisation applied before the classifier.
    """

    def __init__(
        self,
        model: nn.Module,
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        tau: float = 2.0,
        batch_size: int = 128,
        normalize_fn=None,
    ):
        self.model = model
        self.model.eval()
        self.tau = tau
        self.normalize_fn = normalize_fn
        self.device = next(model.parameters()).device

        # Compute class centroids
        self.centroids = self._compute_centroids(
            train_images, train_labels, batch_size
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_centroids(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
    ) -> Dict[int, torch.Tensor]:
        """Compute mean penultimate-feature centroid per class."""
        class_features: Dict[int, List[torch.Tensor]] = {}

        for start in range(0, images.shape[0], batch_size):
            batch = images[start : start + batch_size].to(self.device)
            if self.normalize_fn is not None:
                batch = torch.stack([self.normalize_fn(b) for b in batch])
            feats = extract_penultimate_features(self.model, batch).cpu()
            batch_labels = labels[start : start + batch_size]

            for i, lbl in enumerate(batch_labels.tolist()):
                class_features.setdefault(int(lbl), []).append(feats[i])

        centroids: Dict[int, torch.Tensor] = {}
        for cls, feat_list in class_features.items():
            centroids[cls] = torch.stack(feat_list).mean(dim=0)
        return centroids

    # ------------------------------------------------------------------
    @torch.no_grad()
    def filter_and_weight(
        self,
        views: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute weights for *views* and zero-out OOD ones.

        Parameters
        ----------
        views : Tensor
            ``(V, 3, H, W)`` augmented views in [0, 1].

        Returns
        -------
        weights : Tensor
            ``(V,)`` non-negative weights (0 for discarded views).
        distances : Tensor
            ``(V,)`` distances to nearest centroid.
        """
        batch = views.to(self.device)
        if self.normalize_fn is not None:
            batch = torch.stack([self.normalize_fn(b) for b in batch])

        feats = extract_penultimate_features(self.model, batch).cpu()

        # Stack all centroids
        centroid_keys = sorted(self.centroids.keys())
        centroid_mat = torch.stack([self.centroids[k] for k in centroid_keys])  # (C, D)

        # Min distance to any centroid
        dists = torch.cdist(feats.unsqueeze(0), centroid_mat.unsqueeze(0)).squeeze(0)  # (V, C)
        min_dists = dists.min(dim=1).values  # (V,)

        # Soft weight: w_i = exp(-d_i)
        weights = torch.exp(-min_dists)

        # Hard threshold: discard views beyond τ
        mask = min_dists <= self.tau
        weights = weights * mask.float()

        return weights, min_dists


# ======================================================================
# Soft Voting
# ======================================================================
def soft_vote(
    logits: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted soft voting over an ensemble of logits.

    Parameters
    ----------
    logits : Tensor
        ``(V, num_classes)`` logits from V views.
    weights : Tensor | None
        ``(V,)`` non-negative weights.  Uniform when ``None``.

    Returns
    -------
    Tensor
        ``(1, num_classes)`` aggregated logits.
    """
    if weights is None:
        weights = torch.ones(logits.shape[0], device=logits.device)

    # Avoid division by zero if all weights are 0
    w_sum = weights.sum()
    if w_sum == 0:
        weights = torch.ones_like(weights)
        w_sum = weights.sum()

    probs = F.softmax(logits, dim=1)  # (V, C)
    weighted = (probs * weights.unsqueeze(1)).sum(dim=0, keepdim=True) / w_sum
    return weighted  # (1, C) — soft probabilities
