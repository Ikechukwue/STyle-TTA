from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm

from experiments.data import (
    create_dataset,
    DATASET_SPLITS,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
)
from experiments.tta.augmentation import augment_views
from experiments.tta.constants import DEFAULT_SEED, RETRIEVAL_TTA_METHODS
from experiments.tta.extract_embeddings import embeddings_exist, extract_and_cache
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.utils.reproducibility import random_seed, worker_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-generate and cache augmented/stylized views.",
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--split", type=str, default="test_r")
    p.add_argument(
        "--tta_method",
        type=str,
        default="retristyle",
        choices=[
            "retristyle",
            "color_tta",
            "adain_tta",
            "geometric",
            "color_jitter",
            "rand_augment",
            "trivial_augment",
            "aug_mix",
            "auto_augment",
        ],
    )
    p.add_argument("--color_method", type=str, default=None)
    p.add_argument("--retrieval_strategy", type=str, default="dino")
    p.add_argument("--n_refs", type=int, default=16)
    p.add_argument("--embedding_dir", type=str, default=None)
    p.add_argument(
        "--embedding_model", type=str, default="vit_base_patch16_dinov3.lvd1689m"
    )
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--native_size", type=int, default=512)
    p.add_argument("--n_views", type=int, default=64)
    p.add_argument("--style_batch_size", type=int, default=8)
    p.add_argument("--cache_root", type=str, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--save_format", type=str, default="png", choices=["png", "jpg", "pt"]
    )
    return p

def _normalize_manifest(manifest: Dict, total_samples: int) -> Dict:
    """Migrate legacy manifest using completed_indices to completed_counts."""
    if "completed_counts" in manifest:
        if len(manifest["completed_counts"]) == total_samples:
            return manifest
        else:
            manifest["completed_counts"] = [0] * total_samples
            return manifest


    old_n_refs = manifest.get("config", {}).get("n_refs", 0)
    completed_set = set(manifest.get("completed_indices", []))

    # Assign old_n_refs to previously completed indices, 0 to others
    completed_counts = [
        old_n_refs if i in completed_set else 0 for i in range(total_samples)
    ]

    manifest["completed_counts"] = completed_counts
    manifest.pop("completed_indices", None)
    return manifest

def _load_manifest(manifest_path: Path) -> Dict:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"completed_counts": [], "config": {}}


def _save_manifest(manifest_path: Path, manifest: Dict):
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(manifest_path)


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_name = f"dino_{args.dataset}_{args.split}_s{args.seed}"
    cache_dir = Path(args.cache_root) / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Augmented Image Pre-Generation")
    print("=" * 72)
    print(f"Cache dir     : {cache_dir}")
    print(f"Dataset       : {args.dataset} (split={args.split})")
    print(f"TTA method    : {args.tta_method}")
    print(f"Device        : {device}")

    g = random_seed(seed_value=args.seed)

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
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_seed,
        generator=g,
    )

    manifest_path = cache_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    manifest = _normalize_manifest(manifest, len(test_set))
    completed_counts = manifest["completed_counts"]

    print(f"Total samples : {len(test_set)}")
    print(
        f"Fully done    : {sum(1 for c in completed_counts if c >= args.n_refs)} samples"
    )

    print(f"Total samples : {len(test_set)}")
    print(
        f"Fully done    : {sum(1 for c in completed_counts if c >= args.n_refs)} samples"
    )
    print("=" * 72)

    manifest["config"] = vars(args)
    _save_manifest(manifest_path, manifest)

    retriever = None
    color_transfer_fn = None
    retristyle_infer = None

    if args.tta_method in RETRIEVAL_TTA_METHODS:
        if args.retrieval_strategy == "dino" and args.embedding_dir:
            extract_and_cache(
                dataset_name=args.dataset,
                split=args.split,
                data_path=args.data_path,
                output_dir=args.embedding_dir,
                model_name=args.embedding_model,
                device=device,
            )

        ref_db = build_reference_db(
            dataset=args.dataset,
            data_path=args.data_path,
            input_size=args.native_size,
            seed=args.seed,
            split=args.split,
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

        if args.tta_method == "retristyle":
            from retristyle.infer_style_base import StyleIDMethod

            retristyle_infer = StyleIDMethod()
        elif args.tta_method in ("color_tta", "adain_tta"):
            from experiments.reference_methods.style_transfer_factory import (
                create_color_transfer_method,
            )

            method_name = args.color_method or "adain"
            color_transfer_fn, _ = create_color_transfer_method(method_name)

    total = len(test_set)
    generated = 0
    skipped = 0
    start_time = time.time()

    for idx, (image, label) in enumerate(tqdm(test_loader, desc="Generating")):
        start_view_idx = completed_counts[idx]
        if start_view_idx >= args.n_refs:
            skipped += 1
            continue

        image = image.to(device)
        n_views_to_gen = args.n_refs - start_view_idx

        views = augment_views(
            image,
            tta_method=args.tta_method,
            n_views=args.n_views if args.tta_method not in RETRIEVAL_TTA_METHODS else n_views_to_gen,
            start_idx=start_view_idx,
            end_idx=args.n_refs,
            retriever=retriever,
            n_refs=args.n_refs,
            color_transfer_fn=color_transfer_fn,
            retristyle_infer=retristyle_infer,
            native_size=args.native_size,
            classifier_size=args.input_size,
            input_size=args.input_size,
            dataset=args.dataset,
            style_batch_size=args.style_batch_size,
        )

        sample_dir = cache_dir / f"{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save original image once at start_view_idx == 0
        if start_view_idx == 0:
            orig_img = image.squeeze(0).cpu()
            orig_filename = f"original.{args.save_format}" if args.save_format != "pt" else "original.pt"
            if args.save_format == "pt":
                torch.save(orig_img, sample_dir / orig_filename)
            else:
                save_image(orig_img, sample_dir / orig_filename)

        # Save generated views starting strictly at start_view_idx
        for v_offset in range(views.shape[0]):
            actual_v_idx = start_view_idx + v_offset
            view = views[v_offset].cpu()

            if args.save_format == "pt":
                torch.save(view, sample_dir / f"view_{actual_v_idx:03d}.pt")
            else:
                save_image(
                    view, sample_dir / f"view_{actual_v_idx:03d}.{args.save_format}"
                )

        label_path = sample_dir / "label.txt"
        if not label_path.exists():
            label_val = label.item() if label.numel() == 1 else label.tolist()
            with open(label_path, "w") as f:
                f.write(str(label_val))

        completed_counts[idx] = start_view_idx + views.shape[0]
        generated += 1

        if generated % 100 == 0:
            manifest["completed_counts"] = completed_counts
            _save_manifest(manifest_path, manifest)

    print(f"\nDone: generated={generated}, skipped={skipped}, total={total}")
    print(f"Cache: {cache_dir}")


if __name__ == "__main__":
    main()
