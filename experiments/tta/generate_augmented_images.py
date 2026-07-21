"""
Augmented Image Pre-Generation & Caching
=========================================

Pre-generates stylized views for every test image and saves them to disk.
This avoids repeated style transfer during ablation sweeps, saving
significant HPC compute time.

Directory layout::

    cache_root/
        {dataset}_{split}_{tta_method}_{retrieval}_{n_refs}_seed{seed}/
            00000/              # sample index
                original.png
                view_001.png
                view_002.png
                ...
            00001/
                ...
            manifest.json       # metadata + resume tracking

Usage::

    python -m experiments.tta.generate_augmented_images --dataset imagenet --split test_r --data_path ./data --tta_method retristyle --retrieval_strategy random --n_refs 1 --seed 71397589 --cache_root ./data/augmented_cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm

# ---- project imports --------------------------------------------------------
from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    DATASET_SPLITS,
)
from experiments.utils.reproducibility import random_seed, worker_seed
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.tta.augmentation import augment_views
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from experiments.tta.extract_embeddings import extract_and_cache, embeddings_exist
from experiments.tta.constants import DEFAULT_SEED


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-generate and cache augmented/stylized views.",
    )
    # Dataset
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--split", type=str, default="test_r")

    # TTA method
    p.add_argument("--tta_method", type=str, default="retristyle",
                    choices=["retristyle", "color_tta", "adain_tta",
                             "geometric", "color_jitter", "rand_augment",
                             "trivial_augment", "aug_mix", "auto_augment"])
    p.add_argument("--color_method", type=str, default=None,
                    help="Style transfer method name (for color_tta)")

    # Retrieval
    p.add_argument("--retrieval_strategy", type=str, default="dino")
    p.add_argument("--n_refs", type=int, default=16)
    p.add_argument("--embedding_dir", type=str, default=None)
    p.add_argument("--embedding_model", type=str,
                    default="vit_base_patch16_dinov3.lvd1689m")

    # Sizes
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--native_size", type=int, default=512)
    p.add_argument("--n_views", type=int, default=64)
    p.add_argument("--style_batch_size", type=int, default=8)

    # Output
    p.add_argument("--cache_root", type=str, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_format", type=str, default="png",
                    choices=["png", "jpg", "pt"])

    return p


def _cache_dir_name(args: argparse.Namespace) -> str:
    """Build a unique cache directory name from experiment config."""
    parts = [
        args.dataset, args.split, args.tta_method,
    ]
    if args.tta_method in ("retristyle", "color_tta", "adain_tta"):
        parts.extend([args.retrieval_strategy, f"nr{args.n_refs}"])
    if args.tta_method == "color_tta" and args.color_method:
        parts.append(args.color_method)
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def _load_manifest(manifest_path: Path) -> Dict:
    """Load manifest or return empty dict."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"completed_indices": [], "config": {}}


def _save_manifest(manifest_path: Path, manifest: Dict):
    """Atomically save manifest."""
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(manifest_path)


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- cache directory ----------------------------------------------------
    #cache_name = _cache_dir_name(args)
    cache_name = f"dino_{args.dataset}_{args.split}_s{args.seed}"
    cache_dir = Path(args.cache_root) / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    completed = set(manifest.get("completed_indices", []))

    # Save config to manifest
    manifest["config"] = vars(args)
    _save_manifest(manifest_path, manifest)

    print("=" * 72)
    print("Augmented Image Pre-Generation")
    print("=" * 72)
    print(f"Cache dir     : {cache_dir}")
    print(f"Dataset       : {args.dataset} (split={args.split})")
    print(f"TTA method    : {args.tta_method}")
    print(f"Already done  : {len(completed)} samples")
    print(f"Device        : {device}")
    print("=" * 72)

    # ---- random seed --------------------------------------------------------
    g = random_seed(seed_value=args.seed)

    # ---- test dataset (unnormalized [0, 1]) ---------------------------------
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=args.input_size),
    ])
    test_set = create_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split=args.split,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_seed, generator=g,
    )

    # ---- retriever setup (for style-transfer methods) -----------------------
    retriever = None
    color_transfer_fn = None
    retristyle_infer = None

    from experiments.tta.constants import RETRIEVAL_TTA_METHODS

    if args.tta_method in RETRIEVAL_TTA_METHODS:
        # Extract embeddings if needed
        if args.retrieval_strategy == "dino" and args.embedding_dir:
            from experiments.tta.extract_embeddings import extract_and_cache
            extract_and_cache(
                dataset_name=args.dataset, split=args.split, data_path=args.data_path,
                output_dir=args.embedding_dir,
                model_name=args.embedding_model, device=device,
            )

        # Build reference database + retriever
        ref_db = build_reference_db(
            dataset=args.dataset,
            data_path=args.data_path,
            input_size=args.native_size,
            seed=args.seed,
            split=args.split
        )
        print("Build database")
        retriever = build_retriever(
            strategy=args.retrieval_strategy,
            db=ref_db,
            embedding_dir=args.embedding_dir,
            embedding_model=args.embedding_model,
            dataset=args.dataset,
            device=str(device),
        )

        # Load style transfer method
        if args.tta_method == "retristyle":
            from retristyle.infer_style_base import StyleIDMethod
            retristyle_infer = StyleIDMethod()
        elif args.tta_method in ("color_tta", "adain_tta"):
            from experiments.reference_methods.style_transfer_factory import (
                create_color_transfer_method,
            )
            method_name = args.color_method or "adain"
            color_transfer_fn, _ = create_color_transfer_method(method_name)

    # ---- main generation loop -----------------------------------------------
    total = len(test_set)
    generated = 0
    skipped = 0
    start_time = time.time()

    for idx, (image, label) in enumerate(tqdm(test_loader, desc="Generating")):
        if idx in completed:
            skipped += 1
            continue

        image = image.to(device)

        # Generate views
        views = augment_views(
            image,
            tta_method=args.tta_method,
            n_views=args.n_views if args.tta_method not in RETRIEVAL_TTA_METHODS else args.n_refs + 1,
            retriever=retriever,
            n_refs=args.n_refs,
            color_transfer_fn=color_transfer_fn,
            retristyle_infer=retristyle_infer,
            native_size=args.native_size,
            classifier_size=args.input_size,
            input_size=args.input_size,
            dataset=args.dataset,
            style_batch_size=args.style_batch_size,
        )  # (V, 3, H, W) in [0, 1]

        # Save views to disk
        sample_dir = cache_dir / f"{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for v_idx in range(views.shape[0]):
            view = views[v_idx].cpu()
            if args.save_format == "pt":
                torch.save(view, sample_dir / f"view_{v_idx:03d}.pt")
            else:
                save_image(view, sample_dir / f"view_{v_idx:03d}.{args.save_format}")

        # Save label
        label_val = label.item() if label.numel() == 1 else label.tolist()
        with open(sample_dir / "label.txt", "w") as f:
            f.write(str(label_val))

        # Update manifest
        completed.add(idx)
        generated += 1

        # Periodic checkpoint (every 100 samples)
        if generated % 100 == 0:
            manifest["completed_indices"] = sorted(completed)
            _save_manifest(manifest_path, manifest)

    # Final manifest save
    manifest["completed_indices"] = sorted(completed)
    manifest["total_samples"] = total
    manifest["generation_time_seconds"] = time.time() - start_time
    _save_manifest(manifest_path, manifest)

    print(f"\nDone: generated={generated}, skipped={skipped}, total={total}")
    print(f"Cache: {cache_dir}")


if __name__ == "__main__":
    main()
