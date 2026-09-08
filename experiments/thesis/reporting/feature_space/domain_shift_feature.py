"""
Feature-space domain gap analysis.
====================================
Compares ImageNet-1K splits in embedding space to quantify domain shift.

Comparisons:
  - test_r   vs train@test_r  → actual domain gap  (IN-1K → IN-R)
  - val@test_r vs train@test_r → baseline in-domain gap (noise floor)

Metrics (per class + global):
  - MMD       : distance between class centroids (fast, stable at N=30)
  - Wasserstein: earth mover's distance via Gaussian approximation
  - KL        : KL divergence under Gaussian assumption (symmetric: 0.5*(KL(P||Q)+KL(Q||P)))

Outputs:
  - JSON with per-class and global results for each backbone
  - UMAP plots (one per backbone, colored by domain)

Usage:
    python domain_shift_feature.py \\
        --embedding_dir ./embeddings \\
        --dataset imagenet \\
        --output_dir ./results/feature_domain \\
        --backbones resnet18 densenet121 vit_base_patch16_224 swin_base_patch4_window7_224 dinov2_vitb14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from experiments.thesis.reporting.pixel_dashboard import load_pipeline_dataset,  process_summaries, extract_class_metrics, TTAVisualizer
from config.helpers import load_json


# ============================================================================
# Embedding loading
# ============================================================================

def _model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "_")


def load_embeddings(embedding_dir: Path, dataset: str, model_name: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels for a given backbone + split.
    Returns (embeddings: (N, D), labels: (N,)) as numpy arrays.
    """
    path = embedding_dir / dataset / _model_tag(model_name) / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No embeddings at {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    embs = data["embeddings"].numpy().astype(np.float32)
    labels = data["labels"].numpy().astype(np.int64)
    return embs, labels


def group_by_class(embeddings: np.ndarray, labels: np.ndarray, limit: Optional[int] = None) -> Dict[int, np.ndarray]:
    """Returns {class_id: (N_cls, D)} dict."""
    groups = {}
    all_labels = np.unique(labels)[:limit]
    for cls in all_labels:
        groups[int(cls)] = embeddings[labels == cls]
    return groups


# ============================================================================
# Metrics
# ============================================================================

def mmd_centroid(X: np.ndarray, Y: np.ndarray) -> float:
    """L2 distance between class centroids on the unit hypersphere.
    Fast and stable at small N. Embeddings are assumed L2-normalised.
    """
    return float(np.linalg.norm(X.mean(0) - Y.mean(0)))


def wasserstein_gaussian(X: np.ndarray, Y: np.ndarray) -> float:
    """2-Wasserstein distance under Gaussian approximation (Frechet distance).
    W2^2 = ||mu_X - mu_Y||^2 + trace(Sigma_X + Sigma_Y - 2*(Sigma_X @ Sigma_Y)^0.5)

    Falls back to centroid distance when N < D (degenerate covariance).
    """
    n, d = X.shape
    m = Y.shape[0]

    mu_x, mu_y = X.mean(0), Y.mean(0)
    mean_term = np.sum((mu_x - mu_y) ** 2)

    # Need enough samples for stable covariance estimation
    if min(n, m) < d:
        # Diagonal-only fallback: ignores covariances
        var_x = X.var(0)
        var_y = Y.var(0)
        trace_term = np.sum((np.sqrt(var_x) - np.sqrt(var_y)) ** 2)
        return float(np.sqrt(max(mean_term + trace_term, 0)))

    cov_x = np.cov(X.T)  # (D, D)
    cov_y = np.cov(Y.T)

    # Compute matrix sqrt of (cov_x @ cov_y) via eigendecomposition
    # More stable than scipy sqrtm for near-singular matrices
    product = cov_x @ cov_y
    eigvals, eigvecs = np.linalg.eigh(product)
    eigvals = np.maximum(eigvals, 0)  # numerical safety
    sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    trace_term = np.trace(cov_x) + np.trace(cov_y) - 2 * np.trace(sqrt_product)
    w2_sq = mean_term + trace_term
    return float(np.sqrt(max(w2_sq, 0)))


def kl_gaussian_symmetric(X: np.ndarray, Y: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric KL divergence between diagonal Gaussians fit to X and Y.
    Symmetric KL = 0.5 * (KL(P||Q) + KL(Q||P))

    Uses diagonal covariance (variance per dimension) — more stable at N=30.
    """
    var_x = X.var(0) + eps
    var_y = Y.var(0) + eps
    mu_x, mu_y = X.mean(0), Y.mean(0)
    d = X.shape[1]
    def kl_diag(mu_p, var_p, mu_q, var_q):
        # KL(N(mu_p, diag(var_p)) || N(mu_q, diag(var_q)))
        return 0.5 * np.sum(
            np.log(var_q / var_p)
            + var_p / var_q
            + (mu_p - mu_q) ** 2 / var_q
            - 1
        ) 
    
    return float(0.5 * (kl_diag(mu_x, var_x, mu_y, var_y) + kl_diag(mu_y, var_y, mu_x, var_x)) / d)


def compute_class_metrics(
    groups_A: Dict[int, np.ndarray],
    groups_B: Dict[int, np.ndarray],
) -> Dict[int, Dict[str, float]]:
    """Compute all metrics for every shared class between two splits."""
    common = sorted(set(groups_A.keys()) & set(groups_B.keys()))
    results = {}
    for cls in common:
        X = groups_A[cls]
        Y = groups_B[cls]
        # Balance sample counts
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]
        results[cls] = {
            "mmd":         mmd_centroid(X, Y),
            "wasserstein": wasserstein_gaussian(X, Y),
            "kl_symmetric": kl_gaussian_symmetric(X, Y),
            "n_samples":   n,
        }
    return results

def aggregate(per_class: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute weighted summary statistics (mean/std/min/max) across all classes 
    for each metric, automatically falling back to an unweighted uniform 
    aggregation if 'n_samples' is missing.
    """
    if not per_class:
        return {}
    
    # Identify valid metric keys, ignoring sample metadata
    keys = [k for k in next(iter(per_class.values())).keys() if k != "n_samples"]
    summary = {}
    
    # Extract weights if available, default to uniform 1s if missing
    sample_counts = [v.get("n_samples", 1) for v in per_class.values()]
    weights = np.array(sample_counts, dtype=float)
    total_weight = np.sum(weights)
    
    for key in keys:
        vals = np.array([v[key] for v in per_class.values()], dtype=float)
        
        if total_weight > 0:
            # Weighted Mean
            w_mean = np.sum(vals * weights) / total_weight
            
            # Weighted Variance and Standard Deviation
            w_var = np.sum(weights * (vals - w_mean) ** 2) / total_weight
            w_std = np.sqrt(w_var)
        else:
            w_mean = np.mean(vals)
            w_std = np.std(vals)
            
        summary[f"{key}_mean"] = float(w_mean)
        summary[f"{key}_std"]  = float(w_std)
        summary[f"{key}_min"]  = float(np.min(vals))
        summary[f"{key}_max"]  = float(np.max(vals))
        
    summary["n_classes"] = len(per_class)
    return summary


# ============================================================================
# Net domain shift (gap - baseline)
# ============================================================================

def compute_net_shift(
    domain_cls: Dict[int, Dict[str, float]],
    baseline_cls: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    """Per-class net shift = domain gap metric - baseline metric.
    This isolates genuine domain shift from in-distribution sampling noise.
    """
    common = sorted(set(domain_cls.keys()) & set(baseline_cls.keys()))
    metrics = ["mmd", "wasserstein", "kl_symmetric"]
    results = {}
    for cls in common:
        results[cls] = {
            f"{m}_net": domain_cls[cls][m] - baseline_cls[cls][m]
            for m in metrics
            if m in domain_cls[cls] and m in baseline_cls[cls]
        }
    return results


# ============================================================================
# UMAP visualisation
# ============================================================================

def plot_umap(
    embs_A: np.ndarray, labels_A: np.ndarray, name_A: str,
    embs_B: np.ndarray, labels_B: np.ndarray, name_B: str,
    backbone: str,
    output_path: Path,
    max_points: int = 1500,
    seed: int = 42,
    limit_cls: Optional[int] = None,
):
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        print("  [skip UMAP] install umap-learn and matplotlib")
        return

    def subsample(embs, labels, n, limit_cls):
        if limit_cls:
            unique_classes = np.unique(labels)[:limit_cls]
            mask = np.isin(labels, unique_classes)
            embs = embs[mask]
            labels = labels[mask]
        if len(embs) <= n:
            return embs, labels
        idx = np.random.default_rng(seed).choice(len(embs), n, replace=False)
        return embs[idx], labels[idx]

    per_split = max_points // 2
    embs_A, labels_A = subsample(embs_A, labels_A, per_split, limit_cls)
    embs_B, labels_B = subsample(embs_B, labels_B, per_split, limit_cls)

    all_embs = np.concatenate([embs_A, embs_B], axis=0)
    domain_labels = np.array([0] * len(embs_A) + [1] * len(embs_B))

    print(f"  Running UMAP on {len(all_embs)} points...")
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=30, min_dist=0.1)
    proj = reducer.fit_transform(all_embs)

    proj_A = proj[:len(embs_A)]
    proj_B = proj[len(embs_A):]

    # ── Plot 1: colored by domain ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, color) in enumerate(zip([name_A, name_B], ["#378ADD", "#D85A30"])):
        mask = domain_labels == i
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=color, s=4, alpha=0.4, linewidths=0, label=name)
    ax.set_title(f"UMAP — {backbone} (by domain)", fontsize=13)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, framealpha=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    domain_path = output_path.parent / (output_path.stem + "_domain.png")
    plt.savefig(domain_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {domain_path}")

    # ── Plot 2: colored by class, marker encodes domain ───────────────────
    common_classes = sorted(set(labels_A.tolist()) & set(labels_B.tolist()))
    n_classes = len(common_classes)
    cmap = plt.get_cmap("tab20" if n_classes <= 20 else "gist_rainbow")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(common_classes):
        color = cmap(i / n_classes)
        mask_A = labels_A == cls
        mask_B = labels_B == cls
        if mask_A.any():
            ax.scatter(proj_A[mask_A, 0], proj_A[mask_A, 1],
                       c=[color], marker='o', s=6, alpha=0.5, linewidths=0)
        if mask_B.any():
            ax.scatter(proj_B[mask_B, 0], proj_B[mask_B, 1],
                       c=[color], marker='^', s=6, alpha=0.5, linewidths=0)

    # Legend: just domain markers, not 200 class colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None',
               markersize=6, label=name_A),
        Line2D([0], [0], marker='^', color='gray', linestyle='None',
               markersize=6, label=name_B),
    ]
    ax.legend(handles=legend_elements, framealpha=0.8)
    ax.set_title(f"UMAP — {backbone} (by class)", fontsize=13)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    class_path = output_path.parent / (output_path.stem + "_class.png")
    plt.savefig(class_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {class_path}")

# ============================================================================
# Main analysis
# ============================================================================

def analyse_backbone(
    backbone: str,
    embedding_dir: Path,
    dataset: str,
    split_domain: str,    # e.g. "test_r"
    split_train: str,     # e.g. "train@test_r"
    split_val: str,       # e.g. "val@test_r"
    output_dir: Path,
    run_umap: bool = True,
    limit: Optional[int] = None, 
):
    print(f"\n{'='*60}")
    print(f"Backbone: {backbone}")
    print(f"{'='*60}")

    # Load all three splits
    embs_domain, labels_domain = load_embeddings(embedding_dir, dataset, backbone, split_domain)
    embs_train,  labels_train  = load_embeddings(embedding_dir, dataset, backbone, split_train)
    embs_val,    labels_val    = load_embeddings(embedding_dir, dataset, backbone, split_val)

    print(f"  {split_domain:20s}: {embs_domain.shape}")
    print(f"  {split_train:20s}: {embs_train.shape}")
    print(f"  {split_val:20s}:   {embs_val.shape}")

    # Group by class
    groups_domain = group_by_class(embs_domain, labels_domain, limit)
    groups_train  = group_by_class(embs_train,  labels_train, limit)
    groups_val    = group_by_class(embs_val,    labels_val, limit)

    # Per-class metrics
    print("  Computing domain gap (split vs train)...")
    domain_cls = compute_class_metrics(groups_domain, groups_train)

    print("  Computing baseline gap (val vs train)...")
    baseline_cls = compute_class_metrics(groups_val, groups_train)

    # Net shift
    net_shift_cls = compute_net_shift(domain_cls, baseline_cls)

    # Global aggregations
    results = {
        "backbone": backbone,
        "splits": {
            "domain":   split_domain,
            "train":    split_train,
            "val":      split_val,
        },
        "domain_gap": {
            "global":    aggregate(domain_cls),
            "per_class": {str(k): v for k, v in domain_cls.items()},
        },
        "baseline_gap": {
            "global":    aggregate(baseline_cls),
            "per_class": {str(k): v for k, v in baseline_cls.items()},
        },
        "net_shift": {
            "global":    aggregate({k: v for k, v in net_shift_cls.items()}),
            "per_class": {str(k): v for k, v in net_shift_cls.items()},
        },
    }

    # Save JSON
    out_json = output_dir / f"{backbone}_domain_gap.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved] {out_json}")

    # UMAP
    if run_umap:
        out_umap = output_dir / f"out_{backbone}_umap.png"
        plot_umap(
            embs_domain, labels_domain, split_domain,
            embs_train,  labels_train,  split_train,
            backbone=backbone,
            output_path=out_umap,
            limit_cls=limit, 
        )

        in_umap = output_dir / f"in_{backbone}_umap.png"
        plot_umap(
            embs_val, labels_val, split_val,
            embs_train,  labels_train,  split_train,
            backbone=backbone,
            output_path=in_umap,
            limit_cls=limit,
        )
    # Quick summary print
    dg = results["domain_gap"]["global"]
    bg = results["baseline_gap"]["global"]
    print(f"\n  {'Metric':<20} {'Domain gap':>12} {'Baseline':>12} {'Net shift':>12}")
    print(f"  {'-'*58}")
    for m in ["mmd", "wasserstein", "kl_symmetric"]:
        d = dg.get(f"{m}_mean", float("nan"))
        b = bg.get(f"{m}_mean", float("nan"))
        print(f"  {m:<20} {d:>12.4f} {b:>12.4f} {d-b:>12.4f}")

    return results

def analyse():
    parser = argparse.ArgumentParser(description="Feature-space domain gap analysis")
    parser.add_argument("--embedding_dir", type=str, default="./data/embeddings")
    parser.add_argument("--dataset",       type=str, default="imagenet")
    parser.add_argument("--output_dir",    type=str, default="./results/domain_gap/feature_space")
    parser.add_argument("--split_domain",  type=str, default="test_r_c26",
                        help="The OOD split to evaluate")
    parser.add_argument("--split_train",   type=str, default="train@test_r",
                        help="In-domain training reference split")
    parser.add_argument("--split_val",     type=str, default="val@test_r",
                        help="In-domain validation split (baseline)")
    parser.add_argument("--backbones", nargs="+", default=[
        "resnet18"])
    
    parser.add_argument("--k_style", default=None)

    parser.add_argument("--no_umap", action="store_true",
                        help="Skip UMAP plots")
    parser.add_argument("--limit_cls",type=int, default=None, 
                        help="Set a limit on how many classes should be included")
    args = parser.parse_args()

    split_domain = args.split_domain #if args.limit_cls != 26 else "test_r_c26"
    if not args.k_style is None:
        split_domain += f"_k{args.k_style}"
    output_dir = Path(args.output_dir) / f"{split_domain}"
    if args.limit_cls:
        output_dir = output_dir / f"{str(args.limit_cls)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    base_results = {}
    from config.constants import ALL_CLASSIFIERS
    for backbone in ALL_CLASSIFIERS:
        try:
            result = analyse_backbone(
                backbone=backbone,
                embedding_dir=Path(args.embedding_dir),
                dataset=args.dataset,
                split_domain=split_domain,
                split_train=args.split_train,
                split_val=args.split_val,
                output_dir=output_dir,
                run_umap=not args.no_umap,
                limit=args.limit_cls
            )
            all_results[backbone] = result["domain_gap"]["global"]
            base_results[backbone] = result["baseline_gap"]["global"]
        except FileNotFoundError as e:
            print(f"  [skip] {backbone}: {e}")

    # Cross-backbone summary table
    if all_results:
        summary_path = output_dir / "cross_backbone_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"{'Backbone':<30} | {'MMD':>8} {'W2':>8} {'KL':>8} | {'MMD':>8} {'W2':>8} {'KL':>8}")
        print("-" * 95)
        
        for bb in all_results.keys():
            dg = all_results[bb]
            bg = base_results.get(bb, {})
            
            # Extract absolute values
            abs_mmd = dg.get('mmd_mean', float('nan'))
            abs_w2 = dg.get('wasserstein_mean', float('nan'))
            abs_kl = dg.get('kl_symmetric_mean', float('nan'))
            
            # Compute net gaps to baseline
            diff_mmd = abs_mmd - bg.get('mmd_mean', float('nan'))
            diff_w2 = abs_w2 - bg.get('wasserstein_mean', float('nan'))
            diff_kl = abs_kl - bg.get('kl_symmetric_mean', float('nan'))
            
            # Clamping long backbone strings to maintain crisp column alignments
            bb_short = bb if len(bb) <= 30 else f"...{bb[-27:]}"
            
            print(f"{bb_short:<30} | "
                  f"{abs_mmd:>8.4f} "
                  f"{abs_w2:>8.4f} "
                  f"{abs_kl:>8.4f} | "
                  f"{diff_mmd:>8.4f} "
                  f"{diff_w2:>8.4f} "
                  f"{diff_kl:>8.4f}")
        print("=" * 110)

if __name__ == "__main__":
    analyse()

"""
python -m experiments.thesis.domain_shift_feature \
    --embedding_dir ./embeddings \
    --dataset imagenet \
    --split_domain test_r --no_umap\
    --split_train train@test_r \
    --split_val val@test_r \
    --backbones densenet121 dinov2_vitb14
"""
