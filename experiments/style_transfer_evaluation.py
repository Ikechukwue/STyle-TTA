"""
Style Transfer Method Comparison for Thesis
============================================

Evaluates all 21(+) implemented style transfer methods using the
same protocol as ``color_transfer.py`` but tailored for the thesis.

Protocol:
    - 20 random content images from the target domain (e.g. ImageNet-R)
    - 40 random style images from the source domain (e.g. ImageNet-1k)
    - All 800 = 20 × 40 combinations per method
    - Compute colour transfer quality + content preservation metrics

Usage::

    python -m experiments.style_transfer_evaluation \\
        --data_path /data/local/retristyle/data \\
        --weights_dir /data/local/retristyle/models/style_transfer \\
        --output_dir ./results/style_transfer_eval \\
        --content_dataset imagenet --content_split test_r \\
        --style_dataset imagenet --style_split train \\
        --n_content 20 --n_style 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import save_image
from tqdm import tqdm

from experiments.data import create_dataset
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.reference_methods.style_transfer_factory import create_color_transfer_method
from experiments.metrics.color_metrics import (
    compute_wasserstein_distance,
    compute_histogram_distance,
    compute_color_moment_distance,
    compute_gatys_style_loss,
)
from experiments.metrics.content_metrics import (
    compute_luminance_ssim,
    compute_ssim,
    compute_lpips_distance,
    compute_edge_similarity,
)

# All methods to evaluate — grouped by category
METHODS = {
    "artistic_trained": [
        "adain", "adaattn", "aespanet", "artflow", "cast",
        "efdm", "iecontrast", "mast", "sanet", "styleformer", "stytr2",
    ],
    "artistic_diffusion": [
        "styleid", "diffstyle",
        # "stylessp", "instantstyle", "diffuseit",  # optional: uncomment if weights available
    ],
    "photorealistic": [
        "modflows", "wct2", "deeppreset", "photonas",
    ],
}

# Weight file mapping for training-required methods
WEIGHT_FILES = {
    "adain": "adain.pth",
    "adaattn": "adaattn.pth",
    "aespanet": "aespanet.pth",
    "artflow": "artflow.pth",
    "cast": "cast.pth",
    "efdm": "efdm.pth",
    "iecontrast": "iecontrast.pth",
    "mast": "mast.pth",
    "sanet": "sanet.pth",
    "styleformer": "styleformer.pth",
    "stytr2": "stytr2.pth",
    "diffstyle": "diffstyle.pt",
    "modflows": "modflows.pt",
    "wct2": "wct2.pt",
    "deeppreset": "deeppreset.tar",
    "photonas": "photonas.pth.tar",
}

# Training-free methods (no weights needed)
TRAINING_FREE_METHODS = {"styleid", "stylessp", "instantstyle", "diffuseit"}


def evaluate_single_method(
    method_name: str,
    weights_path: Optional[str],
    content_images: List[torch.Tensor],
    style_images: List[torch.Tensor],
    output_dir: Path,
    input_size: int = 256,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate a single style transfer method on all content×style pairs."""

    print(f"\n  Loading {method_name}...")
    try:
        method, network = create_color_transfer_method(
            method_name=method_name,
            pretrained_weights=weights_path,
        )
    except Exception as e:
        print(f"  FAILED to load {method_name}: {e}")
        return {"error": str(e)}

    # Get native size
    if hasattr(method, "get_native_image_size"):
        native_size = method.get_native_image_size()
    else:
        native_size = input_size

    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Metrics accumulators
    all_metrics: List[Dict[str, float]] = []
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    total = len(content_images) * len(style_images)
    pbar = tqdm(total=total, desc=f"  {method_name}", leave=False)

    for ci, content_img in enumerate(content_images):
        for si, style_img in enumerate(style_images):
            # Resize to native size for inference
            content_native = F.interpolate(
                content_img.unsqueeze(0), size=(native_size, native_size),
                mode="bilinear", align_corners=False,
            ).to(dev)
            style_native = F.interpolate(
                style_img.unsqueeze(0), size=(native_size, native_size),
                mode="bilinear", align_corners=False,
            ).to(dev)

            # Style transfer
            try:
                with torch.no_grad():
                    output = method(content_native, style_native)
                output = output.squeeze(0).clamp(0, 1).cpu()
            except Exception as e:
                pbar.update(1)
                continue

            # Resize output to evaluation size
            if output.shape[-1] != input_size:
                output = F.interpolate(
                    output.unsqueeze(0), size=(input_size, input_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)

            content_eval = F.interpolate(
                content_img.unsqueeze(0), size=(input_size, input_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            style_eval = F.interpolate(
                style_img.unsqueeze(0), size=(input_size, input_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

            # Compute metrics
            m: Dict[str, float] = {}
            try:
                m["wasserstein"] = compute_wasserstein_distance(output, style_eval, "rgb")
            except Exception:
                m["wasserstein"] = None
            try:
                kl, js, chi2, hist_int = compute_histogram_distance(output, style_eval)
                m["histogram_intersection"] = hist_int
            except Exception:
                m["histogram_intersection"] = None
            try:
                m["color_moment_distance"] = compute_color_moment_distance(output, style_eval)
            except Exception:
                m["color_moment_distance"] = None
            try:
                m["ssim"] = compute_ssim(content_eval, output)
            except Exception:
                m["ssim"] = None
            try:
                m["luminance_ssim"] = compute_luminance_ssim(content_eval, output)
            except Exception:
                m["luminance_ssim"] = None
            try:
                m["lpips"] = compute_lpips_distance(content_eval, output)
            except Exception:
                m["lpips"] = None
            try:
                m["edge_similarity"] = compute_edge_similarity(content_eval, output, method="sobel")
            except Exception:
                m["edge_similarity"] = None

            all_metrics.append(m)

            # Save a few example outputs
            if ci < 3 and si < 3:
                save_image(output, method_dir / f"c{ci:02d}_s{si:02d}.png")

            pbar.update(1)

    pbar.close()

    # Clean up GPU memory
    del method, network
    torch.cuda.empty_cache()

    # Aggregate
    if not all_metrics:
        return {"error": "no successful transfers"}

    summary: Dict[str, float] = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if m.get(key) is not None]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    summary["n_successful"] = len(all_metrics)
    summary["n_total"] = total

    # Save per-pair metrics
    with open(method_dir / "per_pair_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    return summary


def run_evaluation(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("Style Transfer Method Evaluation")
    print("="*72)

    # Load content and style images
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=args.input_size),
    ])

    print(f"\nLoading content images ({args.content_dataset}/{args.content_split}, n={args.n_content})...")
    content_ds = create_dataset(
        args.content_dataset, args.data_path, args.content_split,
        transform=transform,
        use_subset=True, subset_size=args.n_content, subset_seed=args.seed,
    )
    content_images = [content_ds[i][0] for i in range(len(content_ds))]

    print(f"Loading style images ({args.style_dataset}/{args.style_split}, n={args.n_style})...")
    style_ds = create_dataset(
        args.style_dataset, args.data_path, args.style_split,
        transform=transform,
        use_subset=True, subset_size=args.n_style, subset_seed=args.seed + 999983,
    )
    style_images = [style_ds[i][0] for i in range(len(style_ds))]

    print(f"Total pairs: {len(content_images)} × {len(style_images)} = {len(content_images) * len(style_images)}")

    # Save example content/style images
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    for i, img in enumerate(content_images[:5]):
        save_image(img, examples_dir / f"content_{i:02d}.png")
    for i, img in enumerate(style_images[:5]):
        save_image(img, examples_dir / f"style_{i:02d}.png")

    # Evaluate all methods
    all_results: Dict[str, Dict] = {}

    for category, method_list in METHODS.items():
        print(f"\n{'─'*60}")
        print(f"Category: {category}")
        print(f"{'─'*60}")

        # Filter by --methods if specified
        if args.methods:
            method_list = [m for m in method_list if m in args.methods]

        for method_name in method_list:
            # Determine weights path
            if method_name in TRAINING_FREE_METHODS:
                weights_path = None
            elif method_name in WEIGHT_FILES:
                weights_path = str(Path(args.weights_dir) / WEIGHT_FILES[method_name])
                if not Path(weights_path).exists():
                    print(f"  SKIP {method_name}: weights not found at {weights_path}")
                    continue
            else:
                weights_path = None

            start = time.time()
            results = evaluate_single_method(
                method_name=method_name,
                weights_path=weights_path,
                content_images=content_images,
                style_images=style_images,
                output_dir=output_dir,
                input_size=args.input_size,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            elapsed = time.time() - start
            results["category"] = category
            results["elapsed_seconds"] = round(elapsed, 1)
            all_results[method_name] = results

            # Print summary
            if "error" not in results:
                print(f"  {method_name}: "
                      f"Wass={results.get('wasserstein_mean', '?'):.4f}, "
                      f"SSIM={results.get('ssim_mean', '?'):.4f}, "
                      f"LPIPS={results.get('lpips_mean', '?'):.4f}, "
                      f"Edge={results.get('edge_similarity_mean', '?'):.4f} "
                      f"({elapsed:.0f}s)")

    # Save combined results
    results_path = output_dir / "all_methods_comparison.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  [saved] {results_path}")

    # Print ranking table
    print(f"\n{'='*72}")
    print("Method Ranking (by style transfer quality × content preservation)")
    print(f"{'='*72}")
    print(f"{'Method':<18} {'Category':<20} {'Wass↓':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'Edge↑':>8}")
    print(f"{'─'*72}")

    ranked = sorted(
        [(n, r) for n, r in all_results.items() if "error" not in r],
        key=lambda x: x[1].get("ssim_mean", 0) - x[1].get("lpips_mean", 1),
        reverse=True,
    )
    for name, r in ranked:
        print(f"{name:<18} {r['category']:<20} "
              f"{r.get('wasserstein_mean', 0):>8.4f} "
              f"{r.get('ssim_mean', 0):>8.4f} "
              f"{r.get('lpips_mean', 0):>8.4f} "
              f"{r.get('edge_similarity_mean', 0):>8.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Style Transfer Method Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--weights_dir", type=str, default="/data/local/retristyle/models/style_transfer")
    p.add_argument("--output_dir", type=str, default="./results/style_transfer_eval")
    p.add_argument("--content_dataset", type=str, default="imagenet")
    p.add_argument("--content_split", type=str, default="test_r")
    p.add_argument("--style_dataset", type=str, default="imagenet")
    p.add_argument("--style_split", type=str, default="train")
    p.add_argument("--n_content", type=int, default=20)
    p.add_argument("--n_style", type=int, default=40)
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="*", default=None,
                   help="Specific methods to evaluate (default: all)")
    return p


def main():
    args = build_parser().parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
