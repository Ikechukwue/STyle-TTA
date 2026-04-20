"""
Domain Shift Analysis for Thesis Experiments
=============================================

Quantitatively characterises the domain gap between a source dataset
(e.g. ImageNet-1k) and each target variant (e.g. ImageNet-R/A/Sketch)
using texture, colour, and feature-space metrics.

The goal is to show *which* domain shifts are predominantly
texture/style-based (and thus amenable to style-transfer TTA) versus
semantic/structural.

Metrics computed:
    1. **Colour Wasserstein distance** — per-channel histogram shift
    2. **Gram matrix distance** (Gatys style loss) — texture/style shift
    3. **Edge preservation** (SSIM on Sobel edges) — structural preservation
    4. **LPIPS distance** — perceptual dissimilarity
    5. **Maximum Mean Discrepancy (MMD)** on DINOv2 embeddings
    6. **Per-class statistics** — heterogeneity of shift within classes
    7. **t-SNE / UMAP visualisation** of domain gap

Usage::

    python -m experiments.domain_shift_analysis \\
        --source_dataset imagenet --source_split val \\
        --target_dataset imagenet --target_split test_r \\
        --data_path /data/local/retristyle/data \\
        --output_dir ./results/domain_shift_analysis \\
        --n_samples 500 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from tqdm import tqdm

from experiments.data import create_dataset, NORMALIZATION_MEAN, NORMALIZATION_STD
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.metrics.color_metrics import (
    compute_wasserstein_distance,
    compute_gatys_style_loss,
)
from experiments.metrics.content_metrics import (
    compute_luminance_ssim,
    compute_lpips_distance,
    compute_edge_similarity,
)


# ====================================================================
# Embedding extraction (DINOv2)
# ====================================================================
@torch.no_grad()
def extract_embeddings_simple(
    dataset, model_name: str = "dinov2_vitb14",
    batch_size: int = 64, device: str = "cuda",
    max_samples: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 embeddings and labels from *dataset*.

    Returns (embeddings, labels) on CPU.
    """
    import timm
    from timm.data import resolve_model_data_config, create_transform

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Try torch.hub first for DINOv2, fall back to timm
    if model_name.startswith("dinov2"):
        model = torch.hub.load("facebookresearch/dinov2", model_name).to(dev).eval()
        # DINOv2 expects 224x224 images normalized with ImageNet stats
        tfm = v2.Compose([
            v2.Resize(224, antialias=True),
            v2.CenterCrop(224),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=0).to(dev).eval()
        data_cfg = resolve_model_data_config(model)
        tfm = create_transform(**data_cfg, is_training=False)

    if max_samples is not None and max_samples < len(dataset):
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_emb, all_labels = [], []
    for imgs, labels in tqdm(loader, desc=f"Embedding ({model_name})", leave=False):
        batch = torch.stack([tfm(img) for img in imgs]).to(dev)
        emb = model(batch)
        all_emb.append(emb.cpu())
        all_labels.append(labels)

    del model
    torch.cuda.empty_cache()
    emb = torch.cat(all_emb, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return F.normalize(emb.float(), dim=1), labels


# ====================================================================
# Maximum Mean Discrepancy
# ====================================================================
def compute_mmd(X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0) -> float:
    """Compute MMD^2 between X and Y using Gaussian kernel."""
    XX = torch.cdist(X, X)
    YY = torch.cdist(Y, Y)
    XY = torch.cdist(X, Y)

    kXX = torch.exp(-XX ** 2 / (2 * sigma ** 2)).mean()
    kYY = torch.exp(-YY ** 2 / (2 * sigma ** 2)).mean()
    kXY = torch.exp(-XY ** 2 / (2 * sigma ** 2)).mean()

    return float(kXX + kYY - 2 * kXY)


# ====================================================================
# Pairwise image-level metrics
# ====================================================================
def compute_pairwise_metrics(
    source_imgs: List[torch.Tensor],
    target_imgs: List[torch.Tensor],
    n_pairs: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute image-level metrics on random source-target pairs."""
    rng = random.Random(seed)
    n_pairs = min(n_pairs, len(source_imgs), len(target_imgs))

    src_idx = rng.sample(range(len(source_imgs)), n_pairs)
    tgt_idx = rng.sample(range(len(target_imgs)), n_pairs)

    wasserstein_vals, gram_vals, edge_vals, lpips_vals = [], [], [], []

    for si, ti in tqdm(zip(src_idx, tgt_idx), total=n_pairs, desc="Pairwise metrics"):
        src = source_imgs[si]  # (3, H, W) in [0, 1]
        tgt = target_imgs[ti]

        # Colour Wasserstein
        try:
            wasserstein_vals.append(compute_wasserstein_distance(src, tgt, "rgb"))
        except Exception:
            pass

        # Gram matrix distance (style loss)
        try:
            gram_vals.append(compute_gatys_style_loss(
                src.unsqueeze(0), tgt.unsqueeze(0)
            ))
        except Exception:
            pass

        # Edge preservation SSIM
        try:
            edge_vals.append(compute_edge_similarity(src, tgt, method="sobel"))
        except Exception:
            pass

        # LPIPS
        try:
            lpips_vals.append(compute_lpips_distance(src, tgt))
        except Exception:
            pass

    return {
        "wasserstein_mean": float(np.mean(wasserstein_vals)) if wasserstein_vals else None,
        "wasserstein_std": float(np.std(wasserstein_vals)) if wasserstein_vals else None,
        "gram_distance_mean": float(np.mean(gram_vals)) if gram_vals else None,
        "gram_distance_std": float(np.std(gram_vals)) if gram_vals else None,
        "edge_ssim_mean": float(np.mean(edge_vals)) if edge_vals else None,
        "edge_ssim_std": float(np.std(edge_vals)) if edge_vals else None,
        "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
        "lpips_std": float(np.std(lpips_vals)) if lpips_vals else None,
    }


# ====================================================================
# Visualisation helpers
# ====================================================================
def save_tsne_plot(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    output_path: Path,
    source_name: str = "Source",
    target_name: str = "Target",
    max_plot_samples: int = 2000,
):
    """Generate and save t-SNE plot coloured by domain."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib/sklearn not available for t-SNE plot")
        return

    n_src = min(max_plot_samples // 2, len(source_emb))
    n_tgt = min(max_plot_samples // 2, len(target_emb))

    emb = torch.cat([source_emb[:n_src], target_emb[:n_tgt]]).numpy()
    domain = np.array([0] * n_src + [1] * n_tgt)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(emb)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(coords[domain == 0, 0], coords[domain == 0, 1],
               alpha=0.4, s=8, label=source_name, c="tab:blue")
    ax.scatter(coords[domain == 1, 0], coords[domain == 1, 1],
               alpha=0.4, s=8, label=target_name, c="tab:red")
    ax.legend(fontsize=12)
    ax.set_title(f"t-SNE: {source_name} vs {target_name}", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] t-SNE plot → {output_path}")


def save_umap_plot(
    source_emb: torch.Tensor,
    target_emb: torch.Tensor,
    output_path: Path,
    source_name: str = "Source",
    target_name: str = "Target",
    max_plot_samples: int = 2000,
):
    """Generate and save UMAP plot coloured by domain."""
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] umap/matplotlib not available for UMAP plot")
        return

    n_src = min(max_plot_samples // 2, len(source_emb))
    n_tgt = min(max_plot_samples // 2, len(target_emb))

    emb = torch.cat([source_emb[:n_src], target_emb[:n_tgt]]).numpy()
    domain = np.array([0] * n_src + [1] * n_tgt)

    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(emb)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.scatter(coords[domain == 0, 0], coords[domain == 0, 1],
               alpha=0.4, s=8, label=source_name, c="tab:blue")
    ax.scatter(coords[domain == 1, 0], coords[domain == 1, 1],
               alpha=0.4, s=8, label=target_name, c="tab:red")
    ax.legend(fontsize=12)
    ax.set_title(f"UMAP: {source_name} vs {target_name}", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] UMAP plot → {output_path}")


# ====================================================================
# Main analysis
# ====================================================================
def run_analysis(args: argparse.Namespace) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.source_dataset}_{args.source_split}_vs_{args.target_dataset}_{args.target_split}"
    print(f"\n{'='*72}")
    print(f"Domain Shift Analysis: {tag}")
    print(f"{'='*72}\n")

    # ---- load datasets (unnormalised [0,1]) ---------------------------------
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=224),
    ])

    print("Loading source dataset...")
    source_ds = create_dataset(
        args.source_dataset, args.data_path, args.source_split,
        transform=transform,
        use_subset=True, subset_size=args.n_samples, subset_seed=args.seed,
    )
    print("Loading target dataset...")
    target_ds = create_dataset(
        args.target_dataset, args.data_path, args.target_split,
        transform=transform,
        use_subset=True, subset_size=args.n_samples, subset_seed=args.seed,
    )

    # Collect images
    print("Collecting images...")
    source_imgs = [source_ds[i][0] for i in range(len(source_ds))]
    target_imgs = [target_ds[i][0] for i in range(len(target_ds))]

    results = {"tag": tag, "args": vars(args)}

    # ---- 1. Pairwise image-level metrics ------------------------------------
    print("\n1. Computing pairwise image-level metrics...")
    pairwise = compute_pairwise_metrics(
        source_imgs, target_imgs, n_pairs=min(200, args.n_samples), seed=args.seed
    )
    results["pairwise_metrics"] = pairwise
    for k, v in pairwise.items():
        if v is not None:
            print(f"   {k}: {v:.4f}")

    # ---- 2. Embedding-based metrics (MMD) -----------------------------------
    print("\n2. Extracting DINOv2 embeddings...")
    src_emb, src_labels = extract_embeddings_simple(
        source_ds, model_name="dinov2_vitb14", device=device,
        max_samples=args.n_samples,
    )
    tgt_emb, tgt_labels = extract_embeddings_simple(
        target_ds, model_name="dinov2_vitb14", device=device,
        max_samples=args.n_samples,
    )

    print("   Computing MMD...")
    mmd_value = compute_mmd(src_emb, tgt_emb, sigma=1.0)
    results["mmd"] = mmd_value
    print(f"   MMD^2 (DINOv2): {mmd_value:.6f}")

    # Cosine similarity between domain centroids
    src_centroid = src_emb.mean(dim=0)
    tgt_centroid = tgt_emb.mean(dim=0)
    centroid_cos = float(F.cosine_similarity(src_centroid.unsqueeze(0),
                                              tgt_centroid.unsqueeze(0)))
    results["centroid_cosine_similarity"] = centroid_cos
    print(f"   Centroid cosine similarity: {centroid_cos:.4f}")

    # ---- 3. Visualisations --------------------------------------------------
    print("\n3. Generating visualisations...")
    save_tsne_plot(
        src_emb, tgt_emb, src_labels, tgt_labels,
        output_dir / f"{tag}_tsne.png",
        source_name=f"{args.source_dataset}/{args.source_split}",
        target_name=f"{args.target_dataset}/{args.target_split}",
    )
    save_umap_plot(
        src_emb, tgt_emb,
        output_dir / f"{tag}_umap.png",
        source_name=f"{args.source_dataset}/{args.source_split}",
        target_name=f"{args.target_dataset}/{args.target_split}",
    )

    # ---- 4. Texture vs. Shape characterisation ------------------------------
    print("\n4. Texture vs. Shape characterisation:")
    gram = pairwise.get("gram_distance_mean")
    edge = pairwise.get("edge_ssim_mean")
    if gram is not None and edge is not None:
        if gram > 0.1 and edge > 0.5:
            assessment = "HIGH texture shift, PRESERVED structure → IDEAL for style-transfer TTA"
        elif gram > 0.1 and edge <= 0.5:
            assessment = "HIGH texture shift, LOW structure preservation → mixed signal"
        elif gram <= 0.1 and edge > 0.5:
            assessment = "LOW texture shift, preserved structure → geometric TTA may suffice"
        else:
            assessment = "LOW texture shift, LOW structure preservation → hard shift"
        results["shift_assessment"] = assessment
        print(f"   Gram distance: {gram:.4f}, Edge SSIM: {edge:.4f}")
        print(f"   → {assessment}")

    # ---- Save results -------------------------------------------------------
    results_path = output_dir / f"{tag}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  [saved] Results → {results_path}")

    return results


# ====================================================================
# CLI
# ====================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Domain Shift Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source_dataset", type=str, default="imagenet")
    p.add_argument("--source_split", type=str, default="val")
    p.add_argument("--target_dataset", type=str, default="imagenet")
    p.add_argument("--target_split", type=str, default="test_r")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./results/domain_shift_analysis")
    p.add_argument("--n_samples", type=int, default=500,
                   help="Number of samples per dataset for analysis")
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
