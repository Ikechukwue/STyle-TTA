"""
RetriStyle Retrieval Sub-package
================================

Structure-aware retrieval engines for selecting training-set references
that are structurally isomorphic to a test image.

Retrievers
----------
- ``RandomRetriever``          — uniform random sampling
- ``BalancedRandomRetriever``  — class-balanced random sampling
- ``MetricRetriever``          — SSIM-on-gradients / Mutual Information
- ``BalancedMetricRetriever``  — MMR + class-balanced metric search
- ``DinoRetriever``            — DINOv2 embeddings + FAISS fast search

Database
--------
- ``ReferenceDatabase``        — lazy-loading image database backed by a Dataset

Utilities
---------
- ``build_retrieval_db``       — offline database builder (see script)
"""

from .reference_db import ReferenceDatabase
from .random_retriever import RandomRetriever
from .balanced_random_retriever import BalancedRandomRetriever
from .metric_retriever import MetricRetriever
from .balanced_metric_retriever import BalancedMetricRetriever
from .dino_retriever import DinoRetriever

__all__ = [
    "ReferenceDatabase",
    "RandomRetriever",
    "BalancedRandomRetriever",
    "MetricRetriever",
    "BalancedMetricRetriever",
    "DinoRetriever",
]
