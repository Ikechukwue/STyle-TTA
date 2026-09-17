"""
KV-Cache utilities for the diffusion module.

Reference: Section 5.1 of the RetriStyle-TTA report.

Pre-computes, saves, loads, and directly injects the K and V tensors from
training-set references into the attention blocks, bypassing the image
encoder for styles at inference time — cutting FLOPs by ~50 %.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from .style_injection import KVCache, StyleInjectionDiffusion


def precompute_kv_cache(
    diffusion: StyleInjectionDiffusion,
    images: torch.Tensor,
    batch_size: int = 1,
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[KVCache]:
    """Pre-compute KV caches for a set of reference images.

    Parameters
    ----------
    diffusion : StyleInjectionDiffusion
        Initialised diffusion module.
    images : Tensor
        Reference images ``(N, 3, H, W)`` in [0, 1].
    batch_size : int
        Currently caches are built one image at a time.
    device : str | None
        Override device.
    verbose : bool
        Show progress bar.

    Returns
    -------
    list[KVCache]
        One cache entry per image.
    """
    caches: List[KVCache] = []
    iterator = range(images.shape[0])
    if verbose:
        iterator = tqdm(iterator, desc="Building KV caches")

    for i in iterator:
        img = images[i : i + 1]
        if device is not None:
            img = img.to(device)
        cache = diffusion.build_kv_cache(img)
        # Move tensors to CPU to save VRAM
        cpu_cache: KVCache = {}
        for layer, timesteps in cache.items():
            cpu_cache[layer] = {}
            for t, (k, v) in timesteps.items():
                cpu_cache[layer][t] = (k.cpu(), v.cpu())
        caches.append(cpu_cache)
    return caches


def save_kv_caches(caches: List[KVCache], path: str) -> None:
    """Serialise KV caches to disk."""
    torch.save(caches, path)


def load_kv_caches(path: str) -> List[KVCache]:
    """Load KV caches from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def move_kv_cache_to_device(cache: KVCache, device: torch.device) -> KVCache:
    """Move a single KV cache to *device*."""
    out: KVCache = {}
    for layer, timesteps in cache.items():
        out[layer] = {}
        for t, (k, v) in timesteps.items():
            out[layer][t] = (k.to(device), v.to(device))
    return out
