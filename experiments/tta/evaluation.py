"""
Evaluation strategies — how augmented views are aggregated into a prediction.

Strategies
----------
- **vanilla** — Simple average of softmax predictions over all views.
- **ZERO**    — Entropy-based confidence filtering + zero-temperature
                majority vote (Farina et al., NeurIPS 2024).- **TPT**     — Confidence-based entropy filtering (bottom γ=10 %) +
                standard-temperature softmax averaging
                (Shu et al., NeurIPS 2022).- **FOODS**   — Weighted ensemble using inverse distance to training-set
                class centroids; OOD views are discarded past threshold τ.
- **TENT**    — Test-Time Entropy minimisation on BatchNorm affine params.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from retristyle.ensemble_utils import FOODSFilter, soft_vote
from experiments.reference_methods.TTA import TENT

from .constants import ZERO_GAMMA, TPT_GAMMA


# ======================================================================
# Vanilla
# ======================================================================
@torch.no_grad()
def eval_vanilla(
    views: torch.Tensor,
    classifier: nn.Module,
    normalize_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Average softmax predictions of all views.  Returns ``(1, C)``."""
    batch = torch.stack([normalize_fn(v) for v in views]) if normalize_fn is not None else views
    logits = classifier(batch)
    probs = F.softmax(logits, dim=1)
    return probs.mean(dim=0, keepdim=True)


# ======================================================================
# ZERO  (Farina et al., NeurIPS 2024)
# ======================================================================
@torch.no_grad()
def eval_zero(
    views: torch.Tensor,
    classifier: nn.Module,
    normalize_fn: Optional[Callable] = None,
    gamma: float = ZERO_GAMMA,
) -> torch.Tensor:
    """Entropy-filtered majority vote.  Returns ``(1, C)``."""
    batch = torch.stack([normalize_fn(v) for v in views]) if normalize_fn is not None else views
    logits = classifier(batch)
    probs = F.softmax(logits, dim=1)

    entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=1)
    k = max(1, int(len(entropy) * gamma))
    _, filt_idx = entropy.topk(k, largest=False)
    logits_filt = logits[filt_idx]

    zero_temp = torch.finfo(logits_filt.dtype).eps
    votes = (logits_filt / zero_temp).softmax(dim=1).sum(dim=0)
    p_bar = votes / votes.sum()
    return p_bar.unsqueeze(0)


# ======================================================================
# TPT  (Shu et al., NeurIPS 2022)
# ======================================================================
@torch.no_grad()
def eval_tpt(
    views: torch.Tensor,
    classifier: nn.Module,
    normalize_fn: Optional[Callable] = None,
    gamma: float = TPT_GAMMA,
) -> torch.Tensor:
    """Confidence-filtered softmax averaging.  Returns ``(1, C)``.

    TPT retains the views whose prediction entropy falls in the
    bottom-γ percentile (lowest entropy ⟹ highest confidence), then
    averages their softmax probabilities at the model's native
    temperature.

    This mirrors the confidence-selection step of Test-Time Prompt Tuning
    (Shu et al., NeurIPS 2022), adapted for non-VLM classifiers where
    prompt tuning is not applicable.
    """
    batch = torch.stack([normalize_fn(v) for v in views]) if normalize_fn is not None else views
    logits = classifier(batch)
    probs = F.softmax(logits, dim=1)

    # Entropy of each view's prediction
    entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=1)

    # Retain bottom γ (lowest entropy = most confident)
    k = max(1, int(len(entropy) * gamma))
    _, filt_idx = entropy.topk(k, largest=False)

    # Average softmax at normal temperature
    probs_filt = probs[filt_idx]
    return probs_filt.mean(dim=0, keepdim=True)


# ======================================================================
# FOODS
# ======================================================================
@torch.no_grad()
def eval_foods(
    views: torch.Tensor,
    classifier: nn.Module,
    normalize_fn: Optional[Callable],
    foods_filter: FOODSFilter,
) -> torch.Tensor:
    """FOODS weighted ensemble.  Returns ``(1, C)``."""
    batch = torch.stack([normalize_fn(v) for v in views]) if normalize_fn is not None else views
    logits = classifier(batch)
    weights, _ = foods_filter.filter_and_weight(views)
    return soft_vote(logits, weights)


# ======================================================================
# TENT
# ======================================================================
#@torch.no_grad()
def eval_tent(
    image: torch.Tensor,
    tent: TENT,
    normalize_fn: Optional[Callable],
) -> torch.Tensor:
    """Run TENT and return ``(1, C)`` softmax probabilities."""
    normed = normalize_fn(image.squeeze(0)).unsqueeze(0) if normalize_fn is not None else image.squeeze(0).unsqueeze(0)
    logits = tent.predict(normed)
    return F.softmax(logits, dim=1)
