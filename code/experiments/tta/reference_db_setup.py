"""
Reference-database construction & retriever factory.

Builds a :class:`ReferenceDatabase` (lazy image loader) from the training
split of a dataset, and instantiates the requested retriever backed by it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from code.experiments.data import create_dataset
from code.experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from code.retristyle.retrieval import (
    ReferenceDatabase,
    RandomRetriever,
    BalancedRandomRetriever,
    MetricRetriever,
    BalancedMetricRetriever,
    DinoRetriever,
)
from code.retristyle.retrieval.dino_retriever import DEFAULT_EMBEDDING_MODEL

from .constants import DEFAULT_SEED
from .extract_embeddings import embeddings_exist, load_cached_embeddings


# ======================================================================
# Build lazy reference database
# ======================================================================
def build_reference_db(
    dataset: str,
    data_path: str,
    input_size: int,
    seed: int = DEFAULT_SEED,
    split: str = None,
    show=False,
) -> ReferenceDatabase:
    """Create a :class:`ReferenceDatabase` from the training split.

    Only labels are extracted at init time — images are loaded lazily.
    """
    if not show:
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            ResizeWhileRetainAspectRatio(size=input_size),
        ])
    else:
        transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(224, antialias=True),
        v2.CenterCrop((224, 224)),
    ]
)
    if split is not None and split not in ["train", "val", "test"]:
        current_split = f"train@{split}"
    else:
        current_split = "train"
    train_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split=current_split,
        transform=transform,
    )
    
    return ReferenceDatabase.from_dataset(
        train_set, seed=seed, 
    )


# ======================================================================
# Materialise images (for components that need the full tensor)
# ======================================================================
def materialise_images(
    db: ReferenceDatabase,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load every image in *db* into a contiguous ``(N, 3, H, W)`` tensor.

    Used by modules that still require all images in memory
    (e.g. ``FOODSFilter``).
    """
    imgs = torch.stack([db.get_image(i) for i in range(len(db))])
    return imgs, db.labels


# ======================================================================
# Retriever factory
# ======================================================================
def build_retriever(
    strategy: str,
    db: ReferenceDatabase,
    *,
    seed: int= DEFAULT_SEED,
    metric_type: str = "ssim",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dir: str | None = None,
    dataset: str | None = None,
    device: str = "cuda",
    eval_split: str | None = None
):
    """Instantiate the requested retriever backed by *db*.

    For the ``dino`` strategy, pre-computed embeddings are loaded from
    *embedding_dir* if available.

    NOTE: Add split to filter for the relevant classes in Imagenet 
    """
    if strategy == "random":
        return RandomRetriever(db=db, seed=seed)
    if strategy == "balanced_random":
        return BalancedRandomRetriever(db=db, seed=seed)
    if strategy == "metric":
        return MetricRetriever(db=db, metric=metric_type)
    if strategy == "balanced_metric":
        return BalancedMetricRetriever(db=db, metric=metric_type)
    if strategy == "dino":
        # Try to load cached embeddings
        embeddings = None
        if embedding_dir is not None and dataset is not None and eval_split is not None:
            if embeddings_exist(embedding_dir, dataset, embedding_model, f"train@{eval_split}"):
                embeddings = load_cached_embeddings(
                    embedding_dir, dataset, embedding_model, f"train@{eval_split}",
                )
        return DinoRetriever(
            db=db,
            model_name=embedding_model,
            device=device,
            embeddings=embeddings,
        )
    raise ValueError(f"Unknown retrieval strategy: {strategy}")
