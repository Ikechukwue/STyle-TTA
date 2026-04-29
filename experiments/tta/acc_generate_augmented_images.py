"""
Augmented Image Pre-Generation & Caching
=========================================

Pre-generates stylized views for every test image and saves them to disk.
This avoids repeated style transfer during ablation sweeps, saving
significant HPC compute time.

Supports multi-GPU distributed generation via Accelerate. Each process
handles a disjoint shard of the dataset; only the main process writes
the manifest.

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

Usage (single GPU)::

    python -m experiments.tta.generate_augmented_images \
        --dataset imagenet --split test_r --data_path ./data \
        --tta_method retristyle --retrieval_strategy random \
        --n_refs 1 --seed 71397589 --cache_root ./data/augmented_cache

Usage (multi-GPU via Accelerate)::

    accelerate launch --config_file configs/gpu_04.yaml \
        -m experiments.tta.generate_augmented_images \
        --dataset imagenet --split test_r --data_path ./data \
        --tta_method retristyle --retrieval_strategy dino \
        --n_refs 16 --seed 71397589 --cache_root ./data/augmented_cache
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm

# ---- project imports --------------------------------------------------------
from experiments.data import (
    create_dataset,
    DATASET_SPLITS,
)
from experiments.utils.reproducibility import random_seed, worker_seed
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.tta.augmentation import augment_views, augment_views_distributed
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from experiments.tta.extract_embeddings import extract_and_cache, embeddings_exist
from experiments.tta.constants import DEFAULT_SEED, RETRIEVAL_TTA_METHODS


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
    parts = [args.dataset, args.split, args.tta_method]
    if args.tta_method in ("retristyle", "color_tta", "adain_tta"):
        parts.extend([args.retrieval_strategy, f"nr{args.n_refs}"])
    if args.tta_method == "color_tta" and args.color_method:
        parts.append(args.color_method)
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


def _load_manifest(manifest_path: Path) -> Dict:
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

    # ---- Accelerate init ----------------------------------------------------
    accelerator = Accelerator()
    device = accelerator.device

    # ---- cache directory (created by all ranks, safe) -----------------------
    cache_name = _cache_dir_name(args)
    cache_dir = Path(args.cache_root) / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.json"

    # Only main process reads/writes the manifest to avoid race conditions
    if accelerator.is_main_process:
        manifest = _load_manifest(manifest_path)
        manifest["config"] = vars(args)
        _save_manifest(manifest_path, manifest)
        completed = set(manifest.get("completed_indices", []))
    else:
        completed = set()

    # Broadcast completed set to all ranks so every rank can skip done samples
    # We serialise as a sorted list in a tensor for broadcast
    if accelerator.num_processes > 1:
        if accelerator.is_main_process:
            completed_list = sorted(completed)
            size_t = torch.tensor([len(completed_list)], device=device)
        else:
            size_t = torch.tensor([0], device=device)

        from accelerate.utils import broadcast
        size_t = broadcast(size_t, from_process=0)
        n_done = size_t.item()

        if n_done > 0:
            if accelerator.is_main_process:
                done_t = torch.tensor(completed_list, dtype=torch.long, device=device)
            else:
                done_t = torch.zeros(n_done, dtype=torch.long, device=device)
            done_t = broadcast(done_t, from_process=0)
            completed = set(done_t.tolist())

    accelerator.print("=" * 72)
    accelerator.print("Augmented Image Pre-Generation")
    accelerator.print("=" * 72)
    accelerator.print(f"Cache dir     : {cache_dir}")
    accelerator.print(f"Dataset       : {args.dataset} (split={args.split})")
    accelerator.print(f"TTA method    : {args.tta_method}")
    accelerator.print(f"World size    : {accelerator.num_processes}")
    accelerator.print(f"Already done  : {len(completed)} samples")
    accelerator.print(f"Device        : {device}")
    accelerator.print("=" * 72)

    # ---- random seed --------------------------------------------------------
    g = random_seed(seed_value=args.seed)

    # ---- test dataset (unnormalized [0, 1]) ---------------------------------
    available_splits = DATASET_SPLITS.get(args.dataset, ["train", "val", "test"])
    eval_split = args.split if args.split in available_splits else available_splits[-1]

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=args.input_size),
    ])
    test_set = create_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split=eval_split,
        transform=test_transform,
    )

    # Distributed sampler: each rank processes a disjoint shard.
    # shuffle=False preserves the original sample indices.
    if accelerator.num_processes > 1:
        sampler = DistributedSampler(
            test_set,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_set, batch_size=1, sampler=sampler,
            num_workers=args.num_workers, worker_init_fn=worker_seed,
        )
    else:
        test_loader = DataLoader(
            test_set, batch_size=1, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=worker_seed, generator=g,
        )

    # ---- embedding extraction (dino retrieval only) -------------------------
    embedding_model = args.embedding_model or "vit_base_patch16_dinov3.lvd1689m"

    if (
        args.tta_method in RETRIEVAL_TTA_METHODS
        and args.retrieval_strategy == "dino"
        and args.embedding_dir is not None
    ):
        if not embeddings_exist(args.embedding_dir, args.dataset, embedding_model, "train"):
            accelerator.print("Extracting training-set embeddings...")
            extract_and_cache(
                dataset_name=args.dataset,
                data_path=args.data_path,
                split=f"train@{eval_split}",
                output_dir=args.embedding_dir,
                model_name=embedding_model,
                input_size=args.input_size,
                device=str(device),
            )
        if not embeddings_exist(args.embedding_dir, args.dataset, embedding_model, eval_split):
            accelerator.print(f"Extracting {eval_split}-set embeddings...")
            extract_and_cache(
                dataset_name=args.dataset,
                data_path=args.data_path,
                split=eval_split,
                output_dir=args.embedding_dir,
                model_name=embedding_model,
                input_size=args.input_size,
                device=str(device),
            )

    # ---- retriever setup (for style-transfer methods) -----------------------
    retriever = None
    color_transfer_fn = None
    retristyle_infer = None

    if args.tta_method in RETRIEVAL_TTA_METHODS:
        ref_db = build_reference_db(
            dataset=args.dataset,
            data_path=args.data_path,
            input_size=args.native_size,
            seed=args.seed,
            split=args.split,
        )
        accelerator.print("Building reference database...")
        retriever = build_retriever(
            strategy=args.retrieval_strategy,
            db=ref_db,
            embedding_dir=args.embedding_dir,
            embedding_model=embedding_model,
            dataset=args.dataset,
            device=str(device),
        )
        accelerator.print(f"  Retriever ready — {len(ref_db)} references")

        if args.tta_method == "retristyle":
            from retristyle.infer_style_base import StyleIDMethod
            accelerator.print("Initialising StyleID diffusion...")
            retristyle_infer = StyleIDMethod()
            accelerator.print("  StyleID ready")

        elif args.tta_method in ("color_tta", "adain_tta"):
            from experiments.reference_methods.style_transfer_factory import (
                create_color_transfer_method,
            )
            method_name = args.color_method or "adain"
            accelerator.print(f"Loading color-transfer method: {method_name}")
            color_transfer_fn, _ = create_color_transfer_method(method_name)

    # ---- effective view count -----------------------------------------------
    if args.tta_method in RETRIEVAL_TTA_METHODS:
        effective_n_views = args.n_refs + 1   # original + n stylised refs
        effective_n_refs = args.n_refs
    else:
        effective_n_views = args.n_views
        effective_n_refs = 0

    # ---- determine whether to use distributed style transfer ----------------
    # For retrieval methods on multi-GPU, shard references across GPUs exactly
    # as run_inference.py does via augment_views_distributed.
    use_distributed_style = (
        args.tta_method in RETRIEVAL_TTA_METHODS
        and accelerator.num_processes > 1
    )

    # ---- main generation loop -----------------------------------------------
    total = len(test_set)
    generated = 0
    skipped = 0
    start_time = time.time()

    # The DistributedSampler gives each rank a shard; the global sample index
    # is recovered from the sampler's indices list so we can name directories
    # consistently across ranks.
    if accelerator.num_processes > 1:
        # sampler.dataset is the full dataset; sampler builds an indices list
        global_indices = list(sampler)  # indices for this rank, in order
    else:
        global_indices = list(range(total))

    pbar = tqdm(
        total=len(global_indices),
        desc=f"[rank {accelerator.process_index}] Generating",
        disable=not accelerator.is_local_main_process,
    )

    for local_step, (image, label) in enumerate(test_loader):
        global_idx = global_indices[local_step]

        if global_idx in completed:
            skipped += 1
            pbar.update(1)
            continue

        image = image.to(device)

        # ---- generate views -------------------------------------------------
        if use_distributed_style:
            # Shard reference retrievals across GPUs; returns full
            # (n_refs+1, 3, H, W) on every rank after gather.
            from retristyle.infer_style_base import StyleIDMethod
            views = augment_views_distributed(
                image,
                args.tta_method,
                effective_n_refs,
                accelerator=accelerator,
                retriever=retriever,
                color_transfer_fn=color_transfer_fn,
                retristyle_infer=retristyle_infer,
                native_size=args.native_size,
                classifier_size=args.input_size,
                style_batch_size=args.style_batch_size,
            )
        else:
            views = augment_views(
                image,
                tta_method=args.tta_method,
                n_views=effective_n_views,
                retriever=retriever,
                n_refs=effective_n_refs,
                color_transfer_fn=color_transfer_fn,
                retristyle_infer=retristyle_infer,
                native_size=args.native_size,
                classifier_size=args.input_size,
                input_size=args.input_size,
                dataset=args.dataset,
                style_batch_size=args.style_batch_size,
            )  # (V, 3, H, W) in [0, 1]

        # ---- save to disk (all ranks write their own samples) ---------------
        # When using augment_views_distributed, every rank has the full view
        # tensor, so only the main process writes to avoid duplicate I/O.
        should_write = (
            accelerator.is_main_process
            if use_distributed_style
            else True   # each rank owns its own disjoint sample shard
        )

        if should_write:
            sample_dir = cache_dir / f"{global_idx:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for v_idx in range(views.shape[0]):
                view = views[v_idx].cpu()
                if args.save_format == "pt":
                    torch.save(view, sample_dir / f"view_{v_idx:03d}.pt")
                else:
                    save_image(view, sample_dir / f"view_{v_idx:03d}.{args.save_format}")

            label_val = label.item() if label.numel() == 1 else label.tolist()
            with open(sample_dir / "label.txt", "w") as f:
                f.write(str(label_val))

        generated += 1
        pbar.update(1)

        # ---- periodic manifest checkpoint (main process only) ---------------
        if accelerator.is_main_process and generated % 100 == 0:
            # Re-load to merge with any progress from other runs
            manifest = _load_manifest(manifest_path)
            manifest_completed = set(manifest.get("completed_indices", []))
            manifest_completed.add(global_idx)
            manifest["completed_indices"] = sorted(manifest_completed)
            _save_manifest(manifest_path, manifest)

    pbar.close()

    # ---- barrier: wait for all ranks before final manifest ------------------
    accelerator.wait_for_everyone()

    # ---- final manifest (main process only) ---------------------------------
    if accelerator.is_main_process:
        # Scan disk to build a definitive completed list (safe after barrier)
        all_done = sorted(
            int(p.name) for p in cache_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        )
        manifest = _load_manifest(manifest_path)
        manifest["completed_indices"] = all_done
        manifest["total_samples"] = total
        manifest["generation_time_seconds"] = round(time.time() - start_time, 2)
        _save_manifest(manifest_path, manifest)

    accelerator.print(
        f"\nDone: generated={generated}, skipped={skipped}, total={total}"
    )
    accelerator.print(f"Cache: {cache_dir}")


if __name__ == "__main__":
    main()
