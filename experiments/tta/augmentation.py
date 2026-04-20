"""
Augmented-view generation for TTA.

Produces ``(V, 3, H, W)`` views of a single test image via:

- **Augmentation-based** — N stochastic augmented copies of the input.
  Includes geometric (ZERO paper: crop + flip), color jitter, AutoAugment,
  RandAugment, TrivialAugment, AugMix, grayscale, random erasing,
  targeted augment, and an *oracle* that randomly picks from all of them.
- **TENT** — Identity (single forward pass).
- **Style-transfer** (color_tta, retristyle, adain_tta) — *n_refs*
  stylised copies produced by passing the test image + retrieved
  reference through a colour- or diffusion-based style transfer.

A *distributed* variant shards the style transfers across GPUs using
Accelerate.
"""

from __future__ import annotations

import random as _random
from typing import List

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast
from torchvision.transforms import v2

from retristyle.infer_style_base import StyleIDMethod
from .constants import AUGMENTATION_TTA_METHODS


# ======================================================================
# Resolution helper
# ======================================================================
def _resize(img: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear resize ``(…, H, W)`` tensor to ``(…, size, size)``."""
    if img.shape[-2] == size and img.shape[-1] == size:
        return img
    if img.dim() == 3:
        return F.interpolate(
            img.unsqueeze(0), size=(size, size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return F.interpolate(
        img, size=(size, size), mode="bilinear", align_corners=False,
    )


# ======================================================================
# Augmentation transform builders
# ======================================================================
def _build_single_augmentation(
    tta_method: str,
    input_size: int,
    dataset: str | None = None,
) -> v2.Transform:
    """Return a *single* torchvision v2 transform for the given TTA method."""
    if tta_method == "geometric":
        # ZERO paper: random resized crop + horizontal flip
        return v2.Compose([
            v2.RandomResizedCrop(size=(input_size, input_size)),
            v2.RandomHorizontalFlip(),
        ])
    elif tta_method == "gray_scale":
        return v2.Grayscale(num_output_channels=3)
    elif tta_method == "color_jitter":
        return v2.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.2,
        )
    elif tta_method == "auto_augment":
        return v2.AutoAugment()
    elif tta_method == "rand_augment":
        return v2.RandAugment()
    elif tta_method == "trivial_augment":
        return v2.TrivialAugmentWide()
    elif tta_method == "aug_mix":
        return v2.AugMix()
    elif tta_method == "random_resized_crop":
        return v2.RandomResizedCrop(size=(input_size, input_size))
    elif tta_method == "random_flip":
        return v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
        ])
    elif tta_method == "random_erasing":
        return v2.RandomErasing()
    elif tta_method == "targeted_augment":
        return _build_targeted_augment(dataset)
    else:
        raise ValueError(
            f"Unknown augmentation TTA method: {tta_method}"
        )


def _build_targeted_augment(dataset: str | None) -> v2.Transform:
    """Build the targeted MedMNIST-C augmentation for the dataset.

    ``AugMedMNISTC`` operates on PIL Images (it accesses ``img.size`` as
    a ``(W, H)`` tuple).  Since our pipeline works with float tensors in
    [0, 1], we wrap the corruption with tensor → PIL → tensor conversions.
    """
    from medmnistc.augmentation import AugMedMNISTC
    from medmnistc.corruptions.registry import CORRUPTIONS_DS

    if dataset is None:
        raise ValueError(
            "dataset is required for targeted_augment"
        )
    if "mnist" in dataset:
        aug = AugMedMNISTC(CORRUPTIONS_DS[dataset])
    elif dataset in ("camelyon17wilds", "epistr"):
        aug = AugMedMNISTC(CORRUPTIONS_DS["pathmnist"])
    elif dataset in ("fitzpatrick17k", "ddi"):
        aug = AugMedMNISTC(CORRUPTIONS_DS["dermamnist"])
    elif dataset in ["peripheral_blood", "bone_marrow_smears_and_peripheral_blood"]:
        aug = AugMedMNISTC(CORRUPTIONS_DS["bloodmnist"])
    elif dataset in ["retina"]:
        aug = AugMedMNISTC(CORRUPTIONS_DS["retinamnist"])
    else:
        raise ValueError(
            f"Targeted augmentation not available for: {dataset}"
        )

    # Wrap: float tensor [0,1] → PIL (uint8) → AugMedMNISTC → tensor [0,1]
    return v2.Compose([
        v2.ToPILImage(),
        aug,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def _build_oracle_pool(
    input_size: int,
    dataset: str | None = None,
) -> List[v2.Transform]:
    """Return a list of all augmentation transforms for oracle TTA."""
    pool: List[v2.Transform] = []
    # All individual augmentation methods except oracle itself
    for method in AUGMENTATION_TTA_METHODS:
        if method == "oracle":
            continue
        try:
            pool.append(
                _build_single_augmentation(method, input_size, dataset)
            )
        except (ValueError, KeyError):
            # targeted_augment may not be available for every dataset
            pass
    return pool


# ======================================================================
# Augmentation-based view generation
# ======================================================================
@torch.no_grad()
def _generate_augmentation_views(
    image: torch.Tensor,      # (1, 3, H, W) [0, 1]
    tta_method: str,
    n_views: int,
    input_size: int,
    dataset: str | None = None,
) -> torch.Tensor:
    """Generate ``(n_views, 3, H, W)`` stochastic augmented views."""
    base = image.squeeze(0)    # (3, H, W)
    views = [base]             # original is always view 0

    if tta_method == "oracle":
        pool = _build_oracle_pool(input_size, dataset)
        for _ in range(n_views - 1):
            aug = _random.choice(pool)
            view = aug(base.clone()).clamp(0, 1).to(base.device)
            views.append(view)
    else:
        aug = _build_single_augmentation(
            tta_method, input_size, dataset,
        )
        for _ in range(n_views - 1):
            view = aug(base.clone()).clamp(0, 1).to(base.device)
            views.append(view)

    return torch.stack(views)  # (n_views, 3, H, W)


# ======================================================================
# Single-process view generation
# ======================================================================
@torch.no_grad()
def augment_views(
    image: torch.Tensor,           # (1, 3, H, W) [0, 1]
    tta_method: str,
    n_views: int,
    *,
    retriever=None,
    n_refs: int = 5,
    color_transfer_fn=None,
    retristyle_infer: StyleIDMethod | None = None,
    native_size: int = 512,
    classifier_size: int = 224,
    input_size: int = 224,
    dataset: str | None = None,
    style_batch_size: int | None = None,
) -> torch.Tensor:
    """Return ``(V, 3, H, W)`` augmented views in [0, 1].

    Parameters
    ----------
    style_batch_size : int | None
        For retrieval-based methods only.  When set, style transfers are
        processed in chunks of this size with intermediate results moved
        to CPU and ``torch.cuda.empty_cache()`` called between chunks.
        This drastically reduces peak VRAM usage so that large view
        counts (e.g. 64) can run on a single GPU.  ``None`` keeps
        everything on GPU (original behaviour).
    """
    device = image.device

    # ---- augmentation-based TTA ---------------------------------------------
    if tta_method in AUGMENTATION_TTA_METHODS:
        return _generate_augmentation_views(
            image, tta_method, n_views,
            input_size=input_size, dataset=dataset,
        )

    # ---- TENT ---------------------------------------------------------------
    if tta_method == "tent":
        return image  # (1, 3, H, W)

    # ---- retrieval-based ----------------------------------------------------
    if retriever is None:
        raise ValueError(
            f"Retriever required for TTA method '{tta_method}'"
        )

    ref_indices, _ = retriever.retrieve(image, k=n_refs)

    # Chunked mode: keep completed views on CPU to save VRAM
    chunked = (
        style_batch_size is not None
        and style_batch_size > 0
    )
    store_device = torch.device("cpu") if chunked else device
    views = [image.squeeze(0).to(store_device)]

    chunk_sz = style_batch_size if chunked else len(ref_indices)
    for chunk_start in range(0, len(ref_indices), chunk_sz):
        chunk_indices = ref_indices[chunk_start:chunk_start + chunk_sz]
        B = len(chunk_indices)

        # -- True batch path for diffusion-based methods -------------------
        # StyleID's __call__ accepts (B, 3, H, W) tensors, so we stack
        # all references in this chunk and process them in a single
        # forward pass (one content inversion + one batched style
        # inversion + one batched sampling).
        if (
            tta_method in ("retristyle", "adain_tta")
            and chunked
            and B > 1
        ):
            if retristyle_infer is None:
                raise ValueError(
                    "retristyle_infer required for retristyle"
                )
            refs = torch.stack(
                [retriever.get_image(idx) for idx in chunk_indices]
            ).to(device)                                        # (B, 3, H, W)
            img_batch = image.expand(B, -1, -1, -1)            # (B, 3, H, W)
            img_native = _resize(img_batch, native_size)        # (B, 3, nat, nat)
            refs_native = _resize(refs, native_size)            # (B, 3, nat, nat)

            out = retristyle_infer(img_native, refs_native)     # (B, 3, H', W')
            out = _resize(out, classifier_size)                 # (B, 3, cls, cls)
            for v in out:
                views.append(v.clamp(0, 1).to(store_device))
            del refs, img_batch, img_native, refs_native, out

        # -- Sequential path (color_tta, single-element chunk, or no
        #    chunking) --------------------------------------------------------
        else:
            for idx in chunk_indices:
                ref = retriever.get_image(idx).unsqueeze(0).to(device)
                img_native = _resize(image, native_size)
                ref_native = _resize(ref, native_size)

                if tta_method == "color_tta":
                    if color_transfer_fn is None:
                        raise ValueError(
                            "color_transfer_fn required for color_tta"
                        )
                    out = color_transfer_fn(img_native, ref_native)
                elif tta_method in ("retristyle", "adain_tta"):
                    if retristyle_infer is None:
                        raise ValueError(
                            "retristyle_infer required for retristyle"
                        )
                    out = retristyle_infer(img_native, ref_native)
                else:
                    raise ValueError(f"Unknown tta_method: {tta_method}")

                out = _resize(out, classifier_size)
                if out.dim() == 4:
                    out = out.squeeze(0)
                views.append(out.clamp(0, 1).to(store_device))
                del ref, out, img_native, ref_native

        if chunked:
            torch.cuda.empty_cache()

    result = torch.stack(views)
    return result.to(device) if chunked else result


# ======================================================================
# Distributed view generation (multi-GPU)
# ======================================================================
@torch.no_grad()
def augment_views_distributed(
    image: torch.Tensor,           # (1, 3, H, W) [0, 1]
    tta_method: str,
    n_refs: int,
    *,
    accelerator: Accelerator,
    retriever=None,
    color_transfer_fn=None,
    retristyle_infer: StyleIDMethod | None = None,
    native_size: int = 512,
    classifier_size: int = 224,
    style_batch_size: int | None = None,
) -> torch.Tensor:
    """Distribute style transfers across GPUs using Accelerate.

    Each process handles a contiguous chunk of reference images.  Within
    each process, references are further batched using
    ``style_batch_size`` for efficient GPU utilisation (true batched
    forward passes through the diffusion model).  Stylised views are
    gathered so every process ends up with the full
    ``(n_refs + 1, 3, H, W)`` tensor.

    Parameters
    ----------
    style_batch_size : int | None
        Number of references to process in a single batched forward pass
        on each GPU.  ``None`` processes references sequentially (original
        behaviour).  Recommended: 4-8 depending on GPU VRAM.
    """
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if retriever is None:
        raise ValueError("Retriever required for distributed augmentation")

    ref_indices, _ = retriever.retrieve(image, k=n_refs)

    if world_size > 1:
        idx_tensor = torch.tensor(ref_indices, dtype=torch.long, device=device)
        idx_tensor = broadcast(idx_tensor, from_process=0)
        ref_indices = idx_tensor.tolist()

    per_gpu = (n_refs + world_size - 1) // world_size
    start = rank * per_gpu
    end = min(start + per_gpu, n_refs)
    my_indices = ref_indices[start:end]

    batched = style_batch_size is not None and style_batch_size > 0
    batch_sz = style_batch_size if batched else max(len(my_indices), 1)

    my_views: List[torch.Tensor] = []
    for batch_start in range(0, len(my_indices), batch_sz):
        batch_indices = my_indices[batch_start:batch_start + batch_sz]
        B = len(batch_indices)

        # -- True batched path for diffusion-based methods -----------------
        if (
            tta_method in ("retristyle", "adain_tta")
            and batched
            and B > 1
        ):
            if retristyle_infer is None:
                raise ValueError(
                    "retristyle_infer required for retristyle/adain_tta"
                )
            refs = torch.stack(
                [retriever.get_image(idx) for idx in batch_indices]
            ).to(device)                                        # (B, 3, H, W)
            img_batch = image.expand(B, -1, -1, -1)            # (B, 3, H, W)
            img_native = _resize(img_batch, native_size)
            refs_native = _resize(refs, native_size)

            out = retristyle_infer(img_native, refs_native)     # (B, 3, H', W')
            out = _resize(out, classifier_size)
            for v in out:
                my_views.append(v.clamp(0, 1))
            del refs, img_batch, img_native, refs_native, out

        # -- Sequential path (color_tta, single-element batch) -------------
        else:
            for idx in batch_indices:
                ref = retriever.get_image(idx).unsqueeze(0).to(device)
                img_native = _resize(image, native_size)
                ref_native = _resize(ref, native_size)

                if tta_method == "color_tta":
                    if color_transfer_fn is None:
                        raise ValueError("color_transfer_fn required for color_tta")
                    out = color_transfer_fn(img_native, ref_native)
                elif tta_method in ("retristyle", "adain_tta"):
                    if retristyle_infer is None:
                        raise ValueError(
                            "retristyle_infer required for retristyle/adain_tta"
                        )
                    out = retristyle_infer(img_native, ref_native)
                else:
                    raise ValueError(
                        f"Unsupported tta_method for distributed: {tta_method}"
                    )

                out = _resize(out, classifier_size)
                if out.dim() == 4:
                    out = out.squeeze(0)
                my_views.append(out.clamp(0, 1))
                del ref, out, img_native, ref_native

        if batched:
            torch.cuda.empty_cache()

    _, C, H, W = image.shape

    if world_size > 1:
        while len(my_views) < per_gpu:
            my_views.append(torch.zeros(C, H, W, device=device))
        local_tensor = torch.stack(my_views)
        all_views = accelerator.gather(local_tensor)
        all_views = all_views[:n_refs]
    else:
        all_views = (
            torch.stack(my_views)
            if my_views
            else torch.empty(0, C, H, W, device=device)
        )

    original = image.squeeze(0).unsqueeze(0)
    return torch.cat([original, all_views], dim=0)
