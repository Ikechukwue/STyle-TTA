"""
Unified TTA Inference — main entry-point.
==========================================

Routes between every TTA flavour (``--tta_method``) and evaluation
strategy (``--eval_strategy``) using a single CLI.

TTA Methods
-----------
* ``geometric``   — 16-view geometric augmentations (crop, flip, rotate)
* ``tent``        — Test-time Entropy minimisation (BatchNorm affine)
* ``color_tta``   — Any of the 21 color/style-transfer reference methods
* ``retristyle``  — RetriStyle diffusion-based TTA (this project)

Evaluation Strategies
---------------------
* ``vanilla`` — Average softmax predictions of all augmented views.
* ``zero``    — ZERO (NeurIPS 2024): Entropy-filter + majority vote.
* ``foods``   — FOODS weighted ensemble with OOD filtering.

Retrieval Strategies (for style-transfer methods)
-------------------------------------------------
* ``random``           — uniform random references
* ``balanced_random``  — class-balanced random references
* ``metric``           — SSIM/MI metric-based retrieval
* ``balanced_metric``  — MMR class-balanced metric retrieval
* ``dino``             — DINOv3 embedding-based retrieval (via FAISS)
* ``base``             — No use of style transfer methods but getting the base results   

Usage
-----
::

    # Quick local test — 1 reference, single GPU
    python -m experiments.tta.run_inference \\
        --dataset pathmnist --data_path ./data \\
        --classifier densenet121 --weights_path ./checkpoints/model.pth \\
        --tta_method retristyle --eval_strategy zero \\
        --retrieval_strategy random --n_refs 1

    python -m experiments.tta.run_inference \
        --dataset imagenet --split test_r --data_path ./data \
        --classifier densenet121 --weights_path ./checkpoints/model.pth \
        --tta_method retristyle --eval_strategy zero \
        --retrieval_strategy random --n_refs 2

        
    # Full ZERO pipeline with DINOv3 retrieval (multi-GPU via Accelerate)
    accelerate launch --config_file configs/gpu_04.yaml \\
        -m experiments.tta.run_inference \\
        --dataset pathmnist --data_path /data \\
        --classifier densenet121 --weights_path /checkpoints/model.pth \\
        --tta_method retristyle --eval_strategy zero \\
        --retrieval_strategy dino --n_refs 64 \\
        --embedding_dir ./embeddings
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Generator
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from accelerate import Accelerator
from tqdm import tqdm
import timm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
import os
os.environ["OUTDATED_IGNORE"] = "1"
# ---- project imports --------------------------------------------------------
from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    TASK_TYPE,
    DATASET_SPLITS,
)
from experiments.utils.reproducibility import random_seed, worker_seed
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.classifier_evaluation import compute_metrics, load_classifier, MaskedClassifier

# TTA baselines
from experiments.reference_methods.TTA import GeometricTTA, TENT

# RetriStyle
from retristyle.infer_style_base import StyleIDMethod
from retristyle.ensemble_utils import FOODSFilter
# Adain 
from experiments.reference_methods.style_transfer.artistic.adain.method import Method as AdaINMethod
# Package-local imports
from .constants import (
    DEFAULT_SEED,
    DEFAULT_N_VIEWS,
    ZERO_N_VIEWS,
    ZERO_GAMMA,
    TPT_GAMMA,
    CACHE_VIEW_CLASSIFIERS,
    AUGMENTATION_TTA_METHODS,
    RETRIEVAL_TTA_METHODS,
    AVAILABLE_TTA_METHODS,
    AVAILABLE_EVAL_STRATEGIES,
    AVAILABLE_RETRIEVAL_STRATEGIES,
)
from .augmentation import augment_views, augment_views_distributed
from .evaluation import eval_vanilla, eval_zero, eval_tpt, eval_foods, eval_tent
from .checkpoint import (
    build_experiment_key,
    predictions_path,
    results_path,
    load_predictions,
    save_predictions,
    save_result,
)
from .reference_db_setup import build_reference_db, materialise_images, build_retriever
from .extract_embeddings import extract_and_cache, embeddings_exist, cache_features


class DummyReferenceDB:
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 0

class DummyRetriever:
    def __init__(self, *args, **kwargs):
        pass

# Color transfer factory — lazy import
def _lazy_create_color_transfer_method(method_name: str, pretrained_weights=None):
    from experiments.reference_methods.style_transfer_factory import (
        create_color_transfer_method,
    )
    return create_color_transfer_method(method_name, pretrained_weights)

def save_generated_views(views: torch.Tensor, cache_root: Path, sample_idx: int):
    """
    Saves live-generated views to the cache directory as full float32 tensors.
    """
    sample_dir = cache_root / f"{sample_idx:05d}"
    
    # Check if this sample already exists to avoid redundant I/O
    
    if sample_dir.exists():
        return
        
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # views shape: (N_views, C, H, W)
    for v_idx in range(views.size(0)):
        view_path = sample_dir / f"view_{v_idx:03d}.pt"
        # Explicitly keep float32 to maintain full data integrity
        torch.save(views[v_idx].detach().cpu().to(torch.float32), view_path)

# ======================================================================
# Normalisation helper
# ======================================================================
def _build_normalize_fn(dataset: str, c_std:tuple, c_mean:tuple) -> Callable[[torch.Tensor], torch.Tensor]:
    mean = torch.tensor(c_mean).view(3, 1, 1)
    std = torch.tensor(c_std).view(3, 1, 1)
    def normalize(img: torch.Tensor) -> torch.Tensor:
        return (img - mean.to(img.device)) / std.to(img.device)

    return normalize


# ======================================================================
# Main inference loop
# ======================================================================
def run_inference(args: argparse.Namespace) -> Dict[str, float]:
    """Execute TTA inference and return metrics dict."""
    accelerator = Accelerator()
    device = accelerator.device

    # ---- dataset info -------------------------------------------------------
    num_classes = NUM_CLASSES[args.dataset]
    task_type = TASK_TYPE[args.dataset]
    available_splits = DATASET_SPLITS.get(args.dataset, ["train", "val", "test"]) 
    eval_split = args.split if args.split in available_splits else available_splits[-1]

    accelerator.print("=" * 72)
    accelerator.print("RetriStyle-TTA — Unified Inference")
    accelerator.print("=" * 72)
    accelerator.print(f"Dataset       : {args.dataset}  (split={eval_split})")
    #accelerator.print(f"Classifier    : {args.classifier}")
    accelerator.print(f"TTA method    : {args.tta_method}")
    accelerator.print(f"Eval strategy : {args.eval_strategy}")
    if args.tta_method in ("adain_tta", "color_tta", "retristyle"):
        accelerator.print(f"Retrieval     : {args.retrieval_strategy}")
        accelerator.print(f"n_refs        : {args.n_refs}")
    if args.tta_method == "color_tta":
        accelerator.print(f"Color method  : {args.color_method}")
    accelerator.print("=" * 72)

    # ---- random seed --------------------------------------------------------
    g = random_seed(seed_value=args.seed)

    # ---- load classifier ----------------------------------------------------
    accelerator.print("Loading classifier...")
    model = load_classifier(
        weights_path=args.weights_path,
        classifier=args.classifier,
        num_classes=num_classes,
        device=device,
    )


    if args.classifier == 'ViT-B-16':
        mean, std = model.preprocess.transforms[-1].mean, model.preprocess.transforms[-1].std
    else:
        mean = NORMALIZATION_MEAN['imagenet']
        std = NORMALIZATION_STD['imagenet']

    if args.split and args.dataset == "imagenet":
        model = MaskedClassifier(model, args.split)

    model = accelerator.prepare(model)
    model.eval()
    normalize_fn = _build_normalize_fn(args.dataset, std, mean)

    # ---- test dataloader (unnormalised [0, 1]) ------------------------------
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
    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_seed, generator=g,
    )

    # ---- embedding extraction (dino retrieval only) -------------------------
    embedding_dir = getattr(args, "embedding_dir", None)
    embedding_model = getattr(args, "embedding_model", None) or "vit_base_patch16_dinov3.lvd1689m"
    augmented_cache_dir = Path(args.augmented_cache) if getattr(args, "augmented_cache", None) else None
    samples_dir = augmented_cache_dir / f"{args.retrieval_strategy}_{args.dataset}_{args.split}_s{str(args.seed)}"
    sample_dir = samples_dir if samples_dir.exists() else None
    
    if (sample_dir is None
        and args.tta_method in ("adain_tta", "color_tta", "retristyle")
        and args.retrieval_strategy == "dino"
        and embedding_dir is not None
    ):
        # Ensure train embeddings exist (extract if not)
        if not embeddings_exist(embedding_dir, args.dataset, embedding_model, "train"):
            accelerator.print("Extracting training-set embeddings...")
            extract_and_cache(
                dataset_name=args.dataset,
                data_path=args.data_path,
                split=f"train@{eval_split}",
                output_dir=embedding_dir,
                model_name=embedding_model,
                input_size=args.input_size,
                device=str(device),
            )
        # Ensure test embeddings exist
        if not embeddings_exist(embedding_dir, args.dataset, embedding_model, eval_split):
            accelerator.print(f"Extracting {eval_split}-set embeddings...")
            extract_and_cache(
                dataset_name=args.dataset,
                data_path=args.data_path,
                split=eval_split,
                output_dir=embedding_dir,
                model_name=embedding_model,
                input_size=args.input_size,
                device=str(device),
            )
        accelerator.print("  Embeddings ready (model can be freed)")

    # ---- retrieval setup (lazy) ---------------------------------------------
    retriever = None
    ref_db = None

    
    if sample_dir is not None:
        ref_db = DummyReferenceDB()
        retriever = DummyRetriever()
        accelerator.print("Using dummy stub components (cache active).")

    if sample_dir is None and args.tta_method in ("adain_tta", "color_tta", "retristyle"):
        accelerator.print("Building reference database (lazy loading)...")
        ref_db = build_reference_db(
            dataset=args.dataset,
            data_path=args.data_path,
            input_size=args.input_size,
            seed=args.seed,
            split=args.split
        )
        print("Fetching Retriever...")
        retriever = build_retriever(
            strategy=args.retrieval_strategy,
            db=ref_db,
            seed = args.seed,
            metric_type=args.metric_type,
            embedding_model=embedding_model,
            embedding_dir=embedding_dir,
            dataset=args.dataset,
            device=str(device),
        )
        accelerator.print(f"  Retriever ready — {len(ref_db)} references")

    # ---- color-transfer function (lazy) -------------------------------------
    color_transfer_fn = None
    if args.tta_method == "color_tta":
        accelerator.print(f"Loading color-transfer method: {args.color_method}")
        color_transfer_fn, ct_model = _lazy_create_color_transfer_method(
            method_name=args.color_method,
            pretrained_weights=args.color_method_weights,
        )
        ct_model = accelerator.prepare(ct_model)
        accelerator.print("  Color-transfer method ready")
    # ---- RetriStyle / StyleID diffusion (lazy) ------------------------------
    retristyle_infer = None
    if args.tta_method == "retristyle":
        accelerator.print("Initialising StyleID diffusion...")
        retristyle_infer = StyleIDMethod()
        accelerator.print("  StyleID diffusion ready")
    # ---- Adain based Styletransfer ------------------------------------------
    if args.tta_method == "adain_tta":
        accelerator.print("Initialising AdaIN...")
        retristyle_infer = AdaINMethod(pretrained_weights=Path(args.method_weights) / "adain.pth")
        accelerator.print("  AdaIN  ready")

    # ---- TENT setup ---------------------------------------------------------
    tent = None
    if args.tta_method == "tent":
        tent = TENT(model=model, lr=args.tent_lr, steps=args.tent_steps,
                    episodic=True, device=device)

    # ---- FOODS filter (lazy) ------------------------------------------------
    foods_filter = None
    if args.eval_strategy == "foods":
        accelerator.print("Building FOODS filter centroids...")
        if ref_db is not None:
            foods_images, foods_labels = materialise_images(ref_db)
        else:
            _foods_db = build_reference_db(
                dataset=args.dataset,
                data_path=args.data_path,
                input_size=args.input_size,
                seed=args.seed,
            )
            foods_images, foods_labels = materialise_images(_foods_db)
        foods_filter = FOODSFilter(
            model=model,
            train_images=foods_images,
            train_labels=foods_labels,
            tau=args.foods_tau,
            batch_size=args.batch_size,
            normalize_fn=normalize_fn,
        )
        accelerator.print("  FOODS filter ready")

    # ---- effective view / ref counts ----------------------------------------
    # n_refs = number of stylised references to generate.
    # effective_n_views = n_refs + 1 (the original test image is always
    #   included as an additional view).
    if args.tta_method in AUGMENTATION_TTA_METHODS:
        effective_n_views = args.n_views
        effective_n_refs = 0  # not used
    elif args.tta_method in RETRIEVAL_TTA_METHODS:
        effective_n_refs = args.n_refs
        effective_n_views = effective_n_refs + 1
    else:
        # tent or any other single-pass method
        effective_n_views = 1
        effective_n_refs = args.n_refs

    # ---- checkpoint / resume ------------------------------------------------
    total_samples = len(test_loader)
    exp_key = build_experiment_key(args, eval_split, args.classifier)
    pred_path = predictions_path(args, eval_split, key=exp_key)
    print(f"Output set to {pred_path}")
    pred_data = load_predictions(pred_path)

    if pred_data.get("completed", False) and len(pred_data["predictions"]) == total_samples:
        accelerator.print(
            f"Experiment already complete ({total_samples} samples). \n"
            f"Predictions at {pred_path} [key={exp_key}]"
        )
        y_true = np.array([p["y_true"] for p in pred_data["predictions"]]).squeeze()
        y_pred = np.array([p["y_pred"] for p in pred_data["predictions"]])
        
        if y_pred.ndim == 3:  # (N, 1, C) from old format
            y_pred = y_pred.squeeze(1)
        num_classes = len(np.unique(test_set.dataset.dataset.targets))
        metrics = compute_metrics(y_true, y_pred, num_classes, task_type)

        res_path = results_path(args, eval_split, key=exp_key)
        if not res_path.exists():         
            result = {
                "dataset": args.dataset,
                "classifier": args.classifier,
                "tta_method": args.tta_method,
                "eval_strategy": args.eval_strategy,
                "retrieval_strategy": getattr(args, "retrieval_strategy", None),
                "color_method": getattr(args, "color_method", None),
                "train_aug": getattr(args, "train_aug", "none"),
                "n_refs": args.n_refs,
                "seed": args.seed,
                "split": eval_split,
                "elapsed_seconds": "afterwards",
                "metrics": metrics,
            }
            save_result(res_path, result)
            print(f"Results were now saved at: {res_path}")
        else:
            print(f"Results are saved at {res_path}")
        return metrics

    start_idx = len(pred_data["predictions"])
    if start_idx > 0:
        accelerator.print(
            f"Resuming from sample {start_idx}/{total_samples} "
            f"({start_idx} already processed)"
        )
    else:
        pred_data["config"] = {
            "dataset": args.dataset,
            #"classifier": args.classifier,
            "tta_method": args.tta_method,
            "eval_strategy": args.eval_strategy,
            "retrieval_strategy": getattr(args, "retrieval_strategy", None),
            "color_method": getattr(args, "color_method", None),
            "train_aug": getattr(args, "train_aug", "none"),
            "n_refs": args.n_refs,
            "n_views": effective_n_views,
            "seed": args.seed,
            "split": eval_split,
        }
        pred_data["total_samples"] = total_samples
        pred_data["completed"] = False

    # ---- augmented image cache -----------------------------------------------
    augmented_cache_dir = None
    if getattr(args, "augmented_cache", None):
        augmented_cache_dir = Path(args.augmented_cache)
        if augmented_cache_dir.exists():
            accelerator.print(f"Using augmented cache: {augmented_cache_dir}")
        else:
            accelerator.print(f"WARNING: Augmented cache not found: {augmented_cache_dir}. \nWill be created")
            #augmented_cache_dir = None

    # ---- inference loop -----------------------------------------------------
    accelerator.print(f"\nStarting inference ({start_idx}/{total_samples} done)...")
    start_t = time.time()

    pbar = tqdm(
        total=total_samples, initial=start_idx, desc="TTA inference",
        disable=not accelerator.is_local_main_process,
    )
    
    for sample_idx, (x, y) in enumerate(test_loader):
        # Skip already-processed samples (resume)
        if sample_idx < start_idx:
            pbar.update(1)
            continue

        x = x.to(device)
        pred = None  # will be set by one of the branches below
        is_feature_cache = False
        # ---- TENT special path ----
        if args.tta_method == "tent":
            pred = eval_tent(x, tent, normalize_fn)
        # ---- Augmented cache path ----
        elif augmented_cache_dir:
         
            sample_dir = augmented_cache_dir / f"{args.retrieval_strategy}_{args.dataset}_{args.split}_s{str(args.seed)}" / f"{sample_idx:05d}"
            if sample_dir.exists():
                cached_views = []
                for vf in sorted(sample_dir.glob("view_*.pt")):
                    if len(cached_views) == args.n_views:
                        break
                    cached_views.append(torch.load(vf, map_location=device, weights_only=True))
                    is_feature_cache = True
                if not cached_views:
                    # Try image files
                    from torchvision.io import read_image
                    from torchvision.transforms.functional import convert_image_dtype
                    for vf in sorted(sample_dir.glob("view_*.png")) + sorted(sample_dir.glob("view_*.jpg")):
                        img = read_image(str(vf))
                        cached_views.append(convert_image_dtype(img, torch.float32))
                        if len(cached_views) == args.n_views:
                            break
                        is_feature_cache = False
                if cached_views:
                    views = torch.stack(cached_views).to(device)

                    # Include original as first view if not already there (feature extractions already have that)
                    #if cached_views[0].ndim == 4:
                    #    views = torch.cat([x, views], dim=0)
                else:
                    views = x  # fallback: just the original

                eff_normalize = None if (is_feature_cache and args.classifier in ["ViT-B-16", "dinov2_vitb14"]) else normalize_fn # To not normalize the extracted features twice 
                
                if args.eval_strategy == "vanilla":
                    pred = eval_vanilla(views, model, eff_normalize)
                elif args.eval_strategy == "zero":
                    pred = eval_zero(views, model, eff_normalize, gamma=args.zero_gamma)
                elif args.eval_strategy == "tpt":
                    pred = eval_tpt(views, model, eff_normalize, gamma=args.tpt_gamma)
                elif args.eval_strategy == "foods":
                    pred = eval_foods(views, model, eff_normalize, foods_filter)
                else:
                    raise ValueError(f"Unknown eval_strategy: {args.eval_strategy}")
            else:
                # Cache miss — fall through to live generation
                pred = None

        # If cache was present but missed, or no cache — generate live
        if augmented_cache_dir is not None and pred is None:
            # Fall through to standard augmentation pipeline
            views = augment_views(
                x, args.tta_method, effective_n_views,
                retriever=retriever,
                n_refs=effective_n_refs,
                color_transfer_fn=color_transfer_fn,
                retristyle_infer=retristyle_infer,
                native_size=args.native_size,
                classifier_size=args.input_size,
                input_size=args.input_size,
                dataset=args.dataset,
                style_batch_size=getattr(args, "style_batch_size", None),
            )
            cache_root = augmented_cache_dir / f"{args.retrieval_strategy}_{args.dataset}_{args.split}_s{str(args.seed)}"
            if args.classifier in ["ViT-B-16", "dinov2_vitb14"]:
                views = cache_features(sample_idx=sample_idx,
                                       views=views,
                                       backbone=model.backbone,
                                       cache_root=cache_root,
                                       normalize_fn=normalize_fn)
                is_feature_cache = True

            elif args.tta_method != "geometric":
                if accelerator.is_main_process:
                    
                    save_generated_views(views, cache_root, sample_idx)
                is_feature_cache = False
                
            eff_normalize = None if (is_feature_cache and args.classifier in ["ViT-B-16", "dinov2_vitb14"]) else normalize_fn
            if args.eval_strategy == "vanilla":
                pred = eval_vanilla(views, model, eff_normalize)
            elif args.eval_strategy == "zero":
                pred = eval_zero(views, model, eff_normalize, gamma=args.zero_gamma)
            elif args.eval_strategy == "tpt":
                pred = eval_tpt(views, model, eff_normalize, gamma=args.tpt_gamma)
            elif args.eval_strategy == "foods":
                pred = eval_foods(views, model, eff_normalize, foods_filter)
            else:
                raise ValueError(f"Unknown eval_strategy: {args.eval_strategy}")

        if augmented_cache_dir is None and args.tta_method != "tent":
            use_distributed = (
                args.tta_method in RETRIEVAL_TTA_METHODS
                and accelerator.num_processes > 1
            )
            if use_distributed:
                views = augment_views_distributed(
                    x, args.tta_method, effective_n_refs,
                    accelerator=accelerator,
                    retriever=retriever,
                    color_transfer_fn=color_transfer_fn,
                    retristyle_infer=retristyle_infer,
                    native_size=args.native_size,
                    classifier_size=args.input_size,
                    style_batch_size=getattr(
                        args, "style_batch_size", None
                    ),
                )
            else:
                views = augment_views(
                    x, args.tta_method, effective_n_views,
                    retriever=retriever,
                    n_refs=effective_n_refs,
                    color_transfer_fn=color_transfer_fn,
                    retristyle_infer=retristyle_infer,
                    native_size=args.native_size,
                    classifier_size=args.input_size,
                    input_size=args.input_size,
                    dataset=args.dataset,
                    style_batch_size=getattr(
                        args, "style_batch_size", None
                    ),
                )

            if args.eval_strategy == "vanilla":
                pred = eval_vanilla(views, model, normalize_fn)
            elif args.eval_strategy == "zero":
                pred = eval_zero(views, model, normalize_fn, gamma=args.zero_gamma)
            elif args.eval_strategy == "tpt":
                pred = eval_tpt(views, model, normalize_fn, gamma=args.tpt_gamma)
            elif args.eval_strategy == "foods":
                pred = eval_foods(views, model, normalize_fn, foods_filter)
            else:
                raise ValueError(f"Unknown eval_strategy: {args.eval_strategy}")

        # ---- persist prediction (main process only) ----
        if accelerator.is_main_process:
            pred_data["predictions"].append({
                "sample_idx": sample_idx,
                "y_true": y.squeeze(0).cpu().numpy().tolist(),   # scalar or (L,) for multi-label
                "y_pred": pred.squeeze(0).detach().cpu().numpy().tolist(),  # (C,)
            })
            if (sample_idx + 1) % 500 == 0 or (sample_idx + 1) == total_samples:
                save_predictions(pred_path, pred_data)

        pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_t
    accelerator.print(f"\nInference complete in {elapsed:.1f}s")

    # ---- mark completed & final flush ---------------------------------------
    if accelerator.is_main_process:
        pred_data["completed"] = True
        pred_data["elapsed_seconds"] = round(elapsed, 2)
        save_predictions(pred_path, pred_data)
        accelerator.print(f"Predictions saved to {pred_path}")

    # ---- compute metrics ----------------------------------------------------
    num_classes = len(np.unique(test_set.dataset.dataset.targets))
    if accelerator.is_main_process:
        y_true = np.array([p["y_true"] for p in pred_data["predictions"]]).squeeze()
        y_pred = np.array([p["y_pred"] for p in pred_data["predictions"]])
        if y_pred.ndim == 3:  # (N, 1, C) from old format
            y_pred = y_pred.squeeze(1)
    else:
        y_true = np.zeros((total_samples,))
        y_pred = np.zeros((total_samples, num_classes))

    metrics = compute_metrics(y_true, y_pred, num_classes, task_type)

    if accelerator.is_main_process:
        accelerator.print(f"\n{'─' * 40}")
        accelerator.print(f"  Accuracy        : {metrics['accuracy']:.4f}")
        accelerator.print(f"  Balanced Acc    : {metrics['balanced_accuracy']:.4f}")
        accelerator.print(f"  AUC             : {metrics['auc']:.4f}")
        accelerator.print(f"  ECE             : {metrics['ece']:.4f}")
        accelerator.print(f"{'─' * 40}")

    # ---- save results summary (per-dataset file) ----------------------------
    if accelerator.is_main_process:
        res_path = results_path(args, eval_split, key=exp_key)
        result = {
            "dataset": args.dataset,
            "classifier": args.classifier,
            "tta_method": args.tta_method,
            "eval_strategy": args.eval_strategy,
            "retrieval_strategy": getattr(args, "retrieval_strategy", None),
            "color_method": getattr(args, "color_method", None),
            "train_aug": getattr(args, "train_aug", "none"),
            "n_refs": args.n_refs,
            "seed": args.seed,
            "split": eval_split,
            "elapsed_seconds": round(elapsed, 2),
            "metrics": metrics,
        }
        save_result(res_path, result)
        accelerator.print(f"Results saved to {res_path}")

    return metrics


# ======================================================================
# CLI
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified TTA Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    #p.add_argument("--classifier_type", type=str, required=True, help="timm model type")
    p.add_argument("--classifier", type=str, required=True, help="timm model name")

    #p.add_argument("--classifier", type=str, required=True, help="timm model name")
    p.add_argument("--weights_path", type=str, default="pretrained", required=True)

    # TTA method
    p.add_argument("--tta_method", type=str, required=True,
                   choices=AVAILABLE_TTA_METHODS)
    p.add_argument("--eval_strategy", type=str, default="zero",
                   choices=AVAILABLE_EVAL_STRATEGIES)
    p.add_argument("--method_weights", type=str, default="./data/models/style_transfer",
                   help="Path to the methods weights")

    # retrieval
    p.add_argument("--retrieval_strategy", type=str, default="random",
                   choices=AVAILABLE_RETRIEVAL_STRATEGIES)
    p.add_argument("--metric_type", type=str, default="ssim", choices=["ssim", "mi"])
    p.add_argument("--n_refs", type=int, default=64,
                   help="Style references per test image")

    # embedding / DINOv3
    p.add_argument("--embedding_dir", type=str, default=None,
                   help="Directory for cached embeddings (dino retrieval)")
    p.add_argument("--embedding_model", type=str, default="vit_base_patch16_dinov3.lvd1689m",
                   help="timm model for embedding extraction")

    # color_tta
    p.add_argument("--color_method", type=str, default=None)
    p.add_argument("--color_method_weights", type=str, default=None)

    # TENT
    p.add_argument("--tent_lr", type=float, default=1e-3)
    p.add_argument("--tent_steps", type=int, default=1)

    # ZERO
    p.add_argument("--zero_gamma", type=float, default=ZERO_GAMMA,
                   help="ZERO: fraction of most-confident views to retain")

    # TPT
    p.add_argument("--tpt_gamma", type=float, default=TPT_GAMMA,
                   help="TPT: fraction of most-confident views to retain (default 0.1 = 10%%)")

    # FOODS
    p.add_argument("--foods_tau", type=float, default=2.0,
                   help="FOODS OOD threshold")

    # augmentation-based TTA
    p.add_argument("--n_views", type=int, default=DEFAULT_N_VIEWS,
                   help="Number of stochastic views for augmentation TTA")
    p.add_argument("--train_aug", type=str, default="none",
                   help="Training augmentation used for the classifier (for key disambiguation)")

    # memory management
    p.add_argument(
        "--style_batch_size", type=int, default=None,
        help=(
            "Process retrieval-based style transfers in chunks of "
            "this size, moving results to CPU between chunks. "
            "Reduces peak VRAM so large n_refs can run on a "
            "single GPU. None = keep all on GPU (default)."
        ),
    )

    # augmented image cache
    p.add_argument(
        "--augmented_cache", type=str, default=None,
        help=(
            "Path to a pre-generated augmented image cache directory "
            "(created by experiments.tta.generate_augmented_images). "
            "When set, views are loaded from disk instead of being "
            "generated on the fly, significantly reducing compute."
        ),
    )

    # general
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--input_size", type=int, default=224,
                   help="Classifier input resolution")
    p.add_argument("--native_size", type=int, default=512,
                   help="Native resolution of the style-transfer method")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--output_path", type=str, default="./results")

    return p


def main():
    args = build_parser().parse_args()
    if args.tta_method == "color_tta" and args.color_method is None:
        raise ValueError("--color_method is required when --tta_method=color_tta")
    run_inference(args)


if __name__ == "__main__":
    main()
