"""
Hybrid TTA: Mixed Geometric + Style-Transfer Augmentation
==========================================================

Generates augmented views that combine geometric and style-transfer
augmentations in configurable ratios.  This allows evaluating whether
a mix of both augmentation families outperforms either alone.

The mixing ratio ``(geo_frac, style_frac)`` specifies how many of the
``n_views`` total views come from each source.  For example, with
``n_views=64`` and ratio ``(0.5, 0.5)``, 32 views are geometric
augmentations and 32 are style-transfer augmentations.

Usage::

    python -m experiments.tta.hybrid \\
        --dataset imagenet --split test_r \\
        --data_path /data/local/retristyle/data \\
        --classifier ViT-B-16 --weights_path pretrained \\
        --retrieval_strategy dino \\
        --n_views 64 --geo_frac 0.5 \\
        --eval_strategy zero \\
        --output_path ./results/hybrid_tta
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

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
from experiments.tta.augmentation import augment_views
from experiments.tta.evaluation import eval_vanilla, eval_zero, eval_tpt
from experiments.tta.checkpoint import (
    build_experiment_key,
    predictions_path,
    load_predictions,
    save_predictions,
    save_result,
    results_path,
)
from experiments.tta.reference_db_setup import build_reference_db, build_retriever
from experiments.tta.extract_embeddings import extract_and_cache, embeddings_exist
from experiments.tta.constants import (
    DEFAULT_SEED,
    ZERO_GAMMA,
    TPT_GAMMA,
    THESIS_HYBRID_RATIOS,
)


def _build_normalize_fn(dataset: str):
    mean = torch.tensor(NORMALIZATION_MEAN[dataset]).view(3, 1, 1)
    std = torch.tensor(NORMALIZATION_STD[dataset]).view(3, 1, 1)

    def normalize(img: torch.Tensor) -> torch.Tensor:
        return (img - mean.to(img.device)) / std.to(img.device)

    return normalize


def generate_hybrid_views(
    sample_idx: int,
    image: torch.Tensor,
    n_views: int,
    geo_frac: float,
    *,
    retriever=None,
    augmented_cache: Path | None = None,
    n_refs: int = 1,
    retristyle_infer=None,
    color_transfer_fn=None,
    native_size: int = 512,
    classifier_size: int = 224,
    input_size: int = 224,
    dataset: str | None = None,
    split: str | None = None,
    style_batch_size: int | None = None,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    
    n_geo = max(0, int(round(n_views * geo_frac)))
    n_style = n_views - n_geo
    
    # Start with original image as view 0
    views_list = [image.squeeze(0)] 

    # 1. Geometric views
    if n_geo > 0:
        geo_views = augment_views(
            image, tta_method="geometric", n_views=n_geo, 
            input_size=input_size, dataset=dataset
        )
        # augment_views includes original, so take index 1 onwards
        if geo_views.shape[0] > 1:
            views_list.append(geo_views[1:])

    # 2. Style transfer views
    if n_style > 0:
        if augmented_cache:
            sample_dir = Path(augmented_cache) / f"dino_{dataset}_{split}_s{str(DEFAULT_SEED)}" / f"{sample_idx:05d}"
            if sample_dir.exists():
                # Get all files, sorted to ensure consistent view order
                all_files = sorted(sample_dir.glob("view_*"))
                # Skip index 0 (original) if it exists as the first cached view
                style_candidates = all_files[1:] if len(all_files) >= 1 else []
                cached_tensors = []
                for vf in style_candidates:
                    if len(cached_tensors) > n_style:
                        break
                    if vf.suffix == ".pt":
                        cached_tensors.append(torch.load(vf, map_location=device, weights_only=True))
                    
                    else:
                        from torchvision.io import read_image
                        from torchvision.transforms.functional import convert_image_dtype
                        img = read_image(str(vf))
                        cached_tensors.append(convert_image_dtype(img, torch.float32))
                
                if cached_tensors:
                    cached_tensors = [t.to(device) for t in cached_tensors]
                    views_list.append(torch.stack(cached_tensors))
        
        elif retriever is not None:
            # Fallback to online augmentation
            tta_method = "retristyle" if retristyle_infer is not None else "color_tta"
            style_views = augment_views(
                image, tta_method=tta_method, n_views=n_style + 1,
                retriever=retriever, n_refs=n_style, 
                color_transfer_fn=color_transfer_fn,
                retristyle_infer=retristyle_infer, native_size=native_size,
                classifier_size=classifier_size, input_size=input_size,
                dataset=dataset, style_batch_size=style_batch_size
            )
            if style_views.shape[0] > 1:
                views_list.append(style_views[1:].to(device))

    # Concatenate all parts
    return torch.cat([v.unsqueeze(0) if v.dim() == 3 else v for v in views_list], dim=0)

def run_hybrid_tta(args: argparse.Namespace) -> Dict[str, float]:
    """Execute hybrid TTA inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = NUM_CLASSES[args.dataset]
    task_type = TASK_TYPE[args.dataset]
    available_splits = DATASET_SPLITS.get(args.dataset, ["train", "val", "test"])
    eval_split = args.split if args.split in available_splits else available_splits[-1]
    augmented_cache = Path(args.augmented_cache) if args.augmented_cache else None
    print("=" * 72)
    print("Hybrid TTA Inference")
    print("=" * 72)
    print(f"Dataset       : {args.dataset} (split={eval_split})")
    print(f"Classifier    : {args.classifier}")
    print(f"Geo fraction  : {args.geo_frac:.2f}")
    print(f"Style fraction: {1 - args.geo_frac:.2f}")
    print(f"n_views       : {args.n_views}")
    print(f"Eval strategy : {args.eval_strategy}")
    print("=" * 72)

    g = random_seed(seed_value=args.seed)

    # Load classifier
    model = load_classifier(
        weights_path=args.weights_path,
        classifier=args.classifier,
        num_classes=num_classes,
        device=device,
    )
    model = MaskedClassifier(model, args.split)
    model.to(device)
    model.eval()
    normalize_fn = _build_normalize_fn(args.dataset)

    # Test dataset
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

    # Retriever setup (for style transfer portion)
    retriever = None
    retristyle_infer = None
    color_transfer_fn = None

    style_frac = 1.0 - args.geo_frac
    if style_frac > 0:
        # Embedding extraction if needed
        if augmented_cache:
            ref_db = None
            retriever = None
        else: 
            embedding_dir = getattr(args, "embedding_dir", None)
            embedding_model = getattr(args, "embedding_model", "vit_base_patch16_dinov3.lvd1689m")

            if args.retrieval_strategy == "dino" and embedding_dir:
                if not embeddings_exist(embedding_dir, args.dataset, embedding_model, f"train@{args.split}.pt"):
                    print("Extracting training-set embeddings...")
                    extract_and_cache(
                        dataset_name=args.dataset,
                        data_path=args.data_path,
                        split="train",
                        output_dir=embedding_dir,
                        model_name=embedding_model,
                        input_size=args.input_size,
                        device=str(device),
                    )

            ref_db = build_reference_db(
                dataset=args.dataset,
                data_path=args.data_path,
                input_size=args.input_size,
                seed=args.seed,
            )
            retriever = build_retriever(
                strategy=args.retrieval_strategy,
                db=ref_db,
                embedding_model=embedding_model,
                embedding_dir=embedding_dir,
                dataset=args.dataset,
                device=str(device),
            )
            print(f"  Retriever ready — {len(ref_db)} references")

        # Load style transfer method
        if args.style_method == "retristyle":
            from retristyle.infer_style_base import StyleIDMethod
            retristyle_infer = StyleIDMethod()
            print("  StyleID ready")
        else:
            from experiments.reference_methods.style_transfer_factory import (
                create_color_transfer_method,
            )
            color_transfer_fn, _ = create_color_transfer_method(args.style_method)
            print(f"  Color transfer ({args.style_method}) ready")

    # Checkpoint setup
    exp_key = f"hybrid_geo{args.geo_frac:.2f}_{args.eval_strategy}_nr{args.n_views + 1}_seed{args.seed}"
    pred_path = Path(args.output_path) / "predictions" / args.dataset / f"{args.dataset}_{args.classifier}_{exp_key}_predictions.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_data = load_predictions(pred_path)

    total_samples = len(test_set)
    start_idx = len(pred_data.get("predictions", []))
    if "predictions" not in pred_data:
        pred_data["predictions"] = []
        pred_data["config"] = vars(args)
        pred_data["completed"] = False

    if pred_data.get("completed", False) and start_idx == total_samples:
        print(f"Already complete ({total_samples} samples).")
        y_true = np.array([p["y_true"] for p in pred_data["predictions"]]).squeeze()
        y_pred = np.array([p["y_pred"] for p in pred_data["predictions"]])
        if y_pred.ndim == 3:
            y_pred = y_pred.squeeze(1)
        return compute_metrics(y_true, y_pred, num_classes, task_type)

    if start_idx > 0:
        print(f"Resuming from sample {start_idx}/{total_samples}")

    # Inference loop
    start_t = time.time()
    n_style_refs = max(1, int(round(args.n_views * (1.0 - args.geo_frac))))

    pbar = tqdm(total=total_samples, initial=start_idx, desc="Hybrid TTA")
    runs = len(test_loader)
    for sample_idx, (x, y) in enumerate(test_loader):
        if sample_idx < start_idx:
            continue

        x = x.to(device)

        views = generate_hybrid_views(
            sample_idx=sample_idx,
            image=x, 
            n_views=args.n_views, 
            geo_frac=args.geo_frac,
            retriever=retriever,
            augmented_cache=augmented_cache, 
            n_refs=n_style_refs,
            retristyle_infer=retristyle_infer,
            color_transfer_fn=color_transfer_fn,
            native_size=args.native_size,
            classifier_size=args.input_size,
            input_size=args.input_size,
            dataset=args.dataset,
            split=args.split, 
            style_batch_size=getattr(args, "style_batch_size", None),
            device=device
        )

        # Evaluate
        if args.eval_strategy == "vanilla":
            pred = eval_vanilla(views, model, normalize_fn)
        elif args.eval_strategy == "zero":
            pred = eval_zero(views, model, normalize_fn, gamma=args.zero_gamma)
        elif args.eval_strategy == "tpt":
            pred = eval_tpt(views, model, normalize_fn, gamma=args.tpt_gamma)
        else:
            raise ValueError(f"Unknown eval strategy: {args.eval_strategy}")

        pred_data["predictions"].append({
            "sample_idx": sample_idx,
            "y_true": y.squeeze(0).cpu().numpy().tolist(),
            "y_pred": pred.squeeze(0).detach().cpu().numpy().tolist(),
        })

        if (sample_idx + 1) % 500 == 0 or (sample_idx + 1 ) == runs:
            save_predictions(pred_path, pred_data)

        pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_t

    pred_data["completed"] = True
    pred_data["elapsed_seconds"] = round(elapsed, 2)
    save_predictions(pred_path, pred_data)

    # Compute metrics
    y_true = np.array([p["y_true"] for p in pred_data["predictions"]]).squeeze()
    y_pred = np.array([p["y_pred"] for p in pred_data["predictions"]])
    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze(1)
    metrics = compute_metrics(y_true, y_pred, num_classes, task_type)

    print(f"\n{'─' * 40}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}")
    print(f"  Balanced Acc    : {metrics['balanced_accuracy']:.4f}")
    print(f"  ECE             : {metrics['ece']:.4f}")
    print(f"{'─' * 40}")

    # Save results
    res_path = Path(args.output_path) / "results" / args.dataset / f"{args.dataset}_{args.classifier}_{exp_key}_results.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": args.dataset,
        "classifier": args.classifier,
        "geo_frac": args.geo_frac,
        "style_frac": 1 - args.geo_frac,
        "n_views": args.n_views,
        "eval_strategy": args.eval_strategy,
        "retrieval_strategy": args.retrieval_strategy,
        "seed": args.seed,
        "split": eval_split,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
    }

    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {res_path}")

    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hybrid TTA: mixed geometric + style-transfer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--classifier", type=str, required=True)
    p.add_argument("--weights_path", type=str, required=True)
    p.add_argument("--augmented_cache", type=str, default=None,
                   help="Path to the directory containing cached stylized views.")
    
    p.add_argument("--geo_frac", type=float, required=True,
                   help="Fraction of views from geometric augmentations (0.0–1.0)")
    p.add_argument("--n_views", type=int, default=64)
    p.add_argument("--n_refs", type=int, default=16)
    
    p.add_argument("--eval_strategy", type=str, default="zero",
                   choices=["vanilla", "zero", "tpt"])
    p.add_argument("--zero_gamma", type=float, default=ZERO_GAMMA)
    p.add_argument("--tpt_gamma", type=float, default=TPT_GAMMA)

    p.add_argument("--retrieval_strategy", type=str, default="dino")
    p.add_argument("--style_method", type=str, default="retristyle",
                   help="Style transfer method for the style portion")
    p.add_argument("--embedding_dir", type=str, default=None)
    p.add_argument("--embedding_model", type=str,
                   default="vit_base_patch16_dinov3.lvd1689m")

    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--native_size", type=int, default=512)
    p.add_argument("--style_batch_size", type=int, default=8)

    p.add_argument("--split", type=str, default="test_r")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_path", type=str, default="./results/hybrid_tta")

    return p


def main():
    args = build_parser().parse_args()
    run_hybrid_tta(args)


if __name__ == "__main__":
    main()
