"""
Standalone embedding extraction for the retrieval database.
============================================================

Computes and caches timm-based embeddings (default: DINOv3 ViT-B/16) for
any dataset split so that the inference pipeline can skip the expensive
forward passes and load pre-computed vectors from disk.

The cache is organised as::

    <output_dir>/<dataset>/<model_tag>/<split>.pt

where ``model_tag`` is the sanitised timm model name (dots → underscores).

Usage
-----
::

    # Extract train + test embeddings for pathmnist
    python -m experiments.tta.extract_embeddings \\
        --dataset pathmnist \\
        --data_path /data \\
        --output_dir ./embeddings \\
        --model_name vit_base_patch16_dinov3.lvd1689m \\
        --splits train test

    # Check whether embeddings already exist (skip if so)
    python -m experiments.tta.extract_embeddings \\
        --dataset pathmnist \\
        --data_path /data \\
        --output_dir ./embeddings \\
        --splits train test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from retristyle.retrieval.dino_retriever import DEFAULT_EMBEDDING_MODEL


# ======================================================================
# Helpers
# ======================================================================
def _model_tag(model_name: str) -> str:
    """Sanitise a timm model name for use in a file path."""
    return model_name.replace("/", "_").replace(".", "_")


def embeddings_path(
    output_dir: str | Path,
    dataset: str,
    model_name: str,
    split: str,
) -> Path:
    """Return the canonical path for a cached embedding file."""
    return Path(output_dir) / dataset / _model_tag(model_name) / f"{split}.pt"


def embeddings_exist(
    output_dir: str | Path,
    dataset: str,
    model_name: str,
    split: str,
) -> bool:
    """Check whether embeddings have already been computed."""
    return embeddings_path(output_dir, dataset, model_name, split).exists()


# ======================================================================
# Core extraction
# ======================================================================
@torch.no_grad()
def extract_embeddings(
    dataset_obj: Dataset,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute ``(N, D)`` normalised embeddings for *dataset_obj*.

    The timm model's own data config (resize + normalisation) is applied
    automatically.

    Parameters
    ----------
    dataset_obj : Dataset
        A PyTorch dataset returning ``(image, label)`` per item.
        Images should be ``(3, H, W)`` tensors in ``[0, 1]``.
    model_name : str
        Any timm-compatible model name.
    batch_size : int
        Batch size for forward passes.
    num_workers : int
        DataLoader workers.
    device : str
        Device for the model.

    Returns
    -------
    Tensor
        ``(N, D)`` L2-normalised embeddings on CPU.
    """
    import timm
    from timm.data import resolve_model_data_config, create_transform
    from experiments.clip_classifier import load_clip_classifier, load_dinov2_classifier
    import torchvision.transforms as T
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if model_name in ["ViT-B-16", "dinov2_vitb14"]:
        if model_name == "ViT-B-16":
            clip_model = load_clip_classifier(model_name=model_name, num_classes=0, device=dev)

            tfm = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                            std=(0.26862954, 0.26130258, 0.27577711))
            ])
        elif model_name == "dinov2_vitb14":

            clip_model = load_dinov2_classifier(num_classes=0, device=dev)
            tfm = T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.Normalize(mean=(0.485, 0.456, 0.406), 
                            std=(0.229, 0.224, 0.225))
            ])
        model = clip_model.backbone.eval()
        emb_tuple = True
    else:
        model = timm.create_model(
            model_name, pretrained=True, num_classes=0,
        ).to(dev).eval()
        data_cfg = resolve_model_data_config(model)
        tfm = create_transform(**data_cfg, is_training=False)
        emb_tuple = False

    loader = DataLoader(
        dataset_obj, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    all_emb: List[torch.Tensor] = []
    all_labels = []
    for imgs, labels in tqdm(loader, desc=f"Embedding ({model_name})", leave=False):
        # imgs: (B, 3, H, W) in [0,1] — apply timm transforms per-image
        batch = torch.stack([tfm(img) for img in imgs]).to(dev)
        res = model(batch)  # (B, D) if ViT its a tuple of (emb, label)
        emb = res if not emb_tuple else res[0]
        all_emb.append(emb.cpu())
        all_labels.append(labels)

    labels = torch.cat(all_labels, dim=0)
    embeddings = torch.cat(all_emb, dim=0)
    embeddings = F.normalize(embeddings.float(), dim=1)

    del model
    torch.cuda.empty_cache()
    return embeddings, labels


def extract_and_cache(
    dataset_name: str,
    data_path: str,
    split: str,
    output_dir: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    input_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Extract embeddings and save to disk.  Returns the cache path.

    If the cache already exists and *force* is False, this is a no-op.
    """
    out = embeddings_path(output_dir, dataset_name, model_name, split)

    if out.exists() and not force:
        print(f"  [skip] {out} already exists")
        return out

    from experiments.data import create_dataset
    from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=input_size),
    ])
    ds = create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split=split,
        transform=transform,
    )

    embs, labels = extract_embeddings(
        ds, model_name=model_name,
        batch_size=batch_size, num_workers=num_workers, device=device,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embs, "labels":labels,"model_name": model_name, "split": split,
                 "dataset": dataset_name, "n": embs.shape[0], "dim": embs.shape[1]}, out)
    print(f"  [saved] {out}  ({embs.shape[0]} × {embs.shape[1]})")
    return out

def cache_features(
    sample_idx: int, 
    views: torch.Tensor, 
    backbone, 
    cache_root: Path,
    normalize_fn
) -> torch.Tensor:
    """Extract features and save them individually to match the loading loop."""
    
    # 1. Create the specific directory for this sample
    sample_dir = cache_root / f"{sample_idx:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    backbone.eval()
    device = next(backbone.parameters()).device

    with torch.no_grad():
        # 2. Normalize and Extract
        x_norm = normalize_fn(views.to(device))
        
        if hasattr(backbone, 'encode_image'):
            features = backbone.encode_image(x_norm)
        else:
            features = backbone(x_norm) # (N_views, D)

        for i, feat in enumerate(features):
            feat_path = sample_dir / f"view_{i:02d}.pt"
            torch.save(feat.cpu(), feat_path)

    return features

def load_cached_embeddings(
    output_dir: str | Path,
    dataset: str,
    model_name: str,
    split: str,
) -> torch.Tensor:
    """Load cached ``(N, D)`` embeddings from disk."""
    path = embeddings_path(output_dir, dataset, model_name, split)
    if not path.exists():
        raise FileNotFoundError(f"No cached embeddings at {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    return data["embeddings"]


# ======================================================================
# CLI
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract & cache timm embeddings for retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./embeddings")
    p.add_argument("--model_name", type=str, default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if cache exists")
    return p


def main():
    args = build_parser().parse_args()
    print(f"Extracting embeddings — model={args.model_name}")
    for split in args.splits:
        print(f"\n[{args.dataset} / {split}]")
        extract_and_cache(
            dataset_name=args.dataset,
            data_path=args.data_path,
            split=split,
            output_dir=args.output_dir,
            model_name=args.model_name,
            input_size=args.input_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            force=args.force,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
