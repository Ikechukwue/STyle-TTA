"""
Result Visualisation for Thesis
================================

Generates publication-quality plots for the thesis from JSON results:

    1. Radar charts comparing style transfer methods
    2. Bar charts for ablation results
    3. Line plots for n_refs sweep
    4. Confusion-matrix style plot for per-sample TTA comparison
    5. t-SNE / UMAP plots (delegated to domain_shift_analysis)
    6. Accuracy vs ECE scatter plots

Usage::

    python -m experiments.reporting.visualize_results \\
        --results_dir ./results \\
        --output_dir ./figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from typing import Dict, List, Optional, Tuple
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _find_json(directory: Path, pattern: str = "*.json") -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def _setup_style():
    """Configure matplotlib for publication-quality plots."""
    if HAS_SNS:
        sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

def _get_names(split:str,
                subset_json: str = "./data/imagenet/imagenet_subsets.json",
                name_json:str = "./data/imagenet/imagenet1k/imagenet_class_index.json" ):
    all_ids = _load_json(name_json)
    name_dict = {}
    for v in all_ids.values():
        name_dict[v[0]] = v[1]

    split_names = []
    sub_ids = _load_json(subset_json)
    split_ids = sub_ids[split]

    split_names = [name_dict[id] for id in split_ids]
    return split_names 



def get_wordnet_taxonomic_order(name_json_path: str = "./data/imagenet/imagenet1k/imagenet_class_index.json") -> list[int]:
    """
    Computes a taxonomic sort order for classes using true WordNet path similarity.
    Groups classes by their lowest common subsumers via hierarchical clustering.
    
    Returns:
        list[int]: A list of class indices sorted by their WordNet hierarchy proximity.
    """
    # Ensure WordNet data is available in the environment
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    # 1. Load the core map to extract WNIDs (e.g., "n02119789")
    try:
        with open(name_json_path) as f:
            class_index_map = json.load(f)
    except Exception as e:
        print(f"Error loading class index map: {e}")
        return []

    # 2. Resolve WNIDs to actual WordNet Synsets
    resolved_synsets = {}
    valid_class_indices = []
    
    for idx_str, (wnid, _) in class_index_map.items():
        idx = int(idx_str)
        try:
            # Parse ImageNet format: n02119789 -> offset 2119789, pos noun ('n')
            offset = int(wnid[1:])
            pos = wnid[0]
            synset = wn.synset_from_pos_and_offset(pos, offset)
            resolved_synsets[idx] = synset
            valid_class_indices.append(idx)
        except Exception:
            # Skip or handle invalid mappings gracefully
            continue

    # Sort indices to establish a deterministic baseline matrix
    valid_class_indices.sort()
    n_classes = len(valid_class_indices)
    
    if n_classes == 0:
        return []

    # 3. Build a Distance Matrix based on WordNet Path Similarity
    # Similarity is bounded (0, 1], where 1.0 means identical synsets.
    # Distance = 1.0 - Similarity
    distance_matrix = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        syn_i = resolved_synsets[valid_class_indices[i]]
        for j in range(i, n_classes):
            syn_j = resolved_synsets[valid_class_indices[j]]
            
            # Compute path similarity (looks at shortest path in hypernym tree)
            sim = syn_i.path_similarity(syn_j)
            if sim is None:
                sim = 0.001 # Fallback minimum connectivity if branches are distinct
                
            dist = 1.0 - sim
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # 4. Perform Hierarchical Clustering to group by lowest mappings
    # Convert square distance matrix to condensed form for scipy linkage
    from scipy.spatial.distance import squareform
    condensed_distances = squareform(distance_matrix)
    
    # Use Ward's minimum variance algorithm to create clean, compact thematic groups
    row_linkage = linkage(condensed_distances, method="ward")
    
    # Extract the optimized leaves order from the tree
    sorted_matrix_indices = leaves_list(row_linkage)
    
    # Map back to your original class integers
    sorted_class_order = [valid_class_indices[i] for i in sorted_matrix_indices]
    
    return sorted_class_order
# =========================================================================
# 1. Style Transfer Method Comparison (Grouped Bar)
# =========================================================================


def plot_style_transfer_comparison(results_dir: Path, output_dir: Path, args):
    """Grouped bar chart comparing style transfer methods."""
    f = results_dir / "style_transfer_eval" / "all_methods_comparison.json"
    if not f.exists():
        print("  [skip] Style transfer comparison not found")
        return

    data = _load_json(f)
    methods, ssim_vals, lpips_vals, edge_vals = [], [], [], []
    for name, m in sorted(data.items()):
        if "error" in m:
            continue
        methods.append(name)
        ssim_vals.append(m.get("ssim_mean", 0))
        lpips_vals.append(m.get("lpips_mean", 0))
        edge_vals.append(m.get("edge_similarity_mean", 0))

    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sort by SSIM for consistent ordering
    idx = np.argsort(ssim_vals)[::-1]
    methods = [methods[i] for i in idx]
    ssim_vals = [ssim_vals[i] for i in idx]
    lpips_vals = [lpips_vals[i] for i in idx]
    edge_vals = [edge_vals[i] for i in idx]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    axes[0].barh(methods, ssim_vals, color=colors)
    axes[0].set_title("SSIM (content) ↑")
    axes[0].invert_yaxis()

    axes[1].barh(methods, lpips_vals, color=colors)
    axes[1].set_title("LPIPS (content) ↓")
    axes[1].invert_yaxis()

    axes[2].barh(methods, edge_vals, color=colors)
    axes[2].set_title("Edge Similarity ↑")
    axes[2].invert_yaxis()

    fig.suptitle("Style Transfer Method Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "style_transfer_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [saved] {out}")


# =========================================================================
# 2. Ablation Bar Charts
# =========================================================================
def plot_ablation_bars(results_dir: Path, output_dir: Path, ablation_type: str, args):
    """Bar chart for a specific ablation axis."""
    abl_dir = results_dir / "ablation"  / "tta_inference" / "results" / "imagenet" / args.split
    if not abl_dir.exists():
        abl_dir = results_dir / "ablation"
    files = _find_json(abl_dir, "*.json")
    if not files:
        print(f"  [skip] No ablation results for {ablation_type}")
        return

    grouped: Dict[str, List[float]] = {}
    for f in files:
        data = _load_json(f)
        if ablation_type == "retrieval":
            key = data.get("retrieval_strategy", "?")
        elif ablation_type == "eval":
            key = data.get("eval_strategy", "?")
        elif ablation_type == "nrefs":
            key = str(data.get("n_refs", "?"))
        else:
            continue
        acc = data.get("metrics", {}).get("accuracy")
        if acc is not None:
            grouped.setdefault(key, []).append(acc * 100)

    if not grouped:
        return

    labels = sorted(grouped.keys(), key=lambda x: np.mean(grouped[x]), reverse=True)
    means = [np.mean(grouped[k]) for k in labels]
    stds = [np.std(grouped[k]) for k in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=plt.cm.tab10(np.arange(len(labels)) % 10))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{ablation_type.capitalize()} Strategy Ablation")
    ax.set_ylim(bottom=max(0, min(means) - 10))

    # Value labels
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out = output_dir / f"ablation_{ablation_type}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [saved] {out}")


# =========================================================================
# 3. n_refs Line Plot
# =========================================================================
def plot_nrefs_sweep(results_dir: Path, output_dir: Path, args):
    """Line plot of accuracy vs n_refs per classifier."""
    abl_dir = results_dir / "ablation" / args.split / "tta_inference" / "results"
    if not abl_dir.exists():
        abl_dir = results_dir / "ablation"
    files = _find_json(abl_dir, "*.json")
    if not files:
        print("  [skip] No n_refs sweep results")
        return

    grouped: Dict[Tuple[int, str], List[float]] = {}
    for f in files:
        data = _load_json(f)
        nr = data.get("n_refs")
        clf = data.get("classifier", "?")
        acc = data.get("metrics", {}).get("accuracy")
        if nr is not None and acc is not None:
            grouped.setdefault((nr, clf), []).append(acc * 100)

    if not grouped:
        return

    classifiers = sorted(set(k[1] for k in grouped))
    nrefs = sorted(set(k[0] for k in grouped))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, clf in enumerate(classifiers):
        means = [np.mean(grouped.get((nr, clf), [0])) for nr in nrefs]
        stds = [np.std(grouped.get((nr, clf), [0])) for nr in nrefs]
        ax.errorbar(nrefs, means, yerr=stds, marker="o", label=clf, capsize=3)

    ax.set_xlabel("Number of Style References ($n_{refs}$)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Style-Transfer TTA: $n_{refs}$ Sweep")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(nrefs)

    fig.tight_layout()
    out = output_dir / "nrefs_sweep.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [saved] {out}")


# =========================================================================
# 4. Hybrid TTA Mixing Ratios
# =========================================================================
def plot_hybrid_tta(results_dir: Path, output_dir: Path, args):
    """Line plot of accuracy across geo/style mixing ratios."""
    hybrid_dir = results_dir / "hybrid_tta"
    if not hybrid_dir.exists():
        hybrid_dir = results_dir / "hybrid_tta"
    files = _find_json(hybrid_dir, "*_results.json")
    if not files:
        print("  [skip] No hybrid TTA results")
        return

    grouped: Dict[Tuple[float, str], List[float]] = {}
    for f in files:
        data = _load_json(f)
        geo = data.get("geo_frac")
        clf = data.get("classifier", "?")
        acc = data.get("metrics", {}).get("accuracy")
        if geo is not None and acc is not None:
            grouped.setdefault((geo, clf), []).append(acc * 100)

    if not grouped:
        return

    classifiers = sorted(set(k[1] for k in grouped))
    ratios = sorted(set(k[0] for k in grouped))

    fig, ax = plt.subplots(figsize=(8, 5))
    for clf in classifiers:
        means = [np.mean(grouped.get((r, clf), [0])) for r in ratios]
        ax.plot([1 - r for r in ratios], means, marker="s", label=clf)

    ax.set_xlabel("Style Transfer Fraction")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hybrid TTA: Geometric vs Style-Transfer Mix")
    ax.legend(fontsize=8)
    ax.set_xticks([1 - r for r in ratios])
    ax.set_xticklabels([f"{(1-r)*100:.0f}%" for r in ratios])

    fig.tight_layout()
    out = output_dir / "hybrid_tta_ratios.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [saved] {out}")


# =========================================================================
# 5. Accuracy vs ECE Scatter
# =========================================================================
def plot_accuracy_vs_ece(results_dir: Path, output_dir: Path, args):
    """Scatter plot of accuracy vs ECE across all experiments."""
    all_points = []
    for pattern in ["thesis/geometric_tta", "thesis/ablation", "thesis/hybrid_tta",
                    "geometric_tta", "ablation", "hybrid_tta"]:
        d = results_dir / pattern / f"{args.split}/tta_inference/results"
        for f in _find_json(d, "*.json"):
            data = _load_json(f)
            m = data.get("metrics", {})
            if "accuracy" in m and "ece" in m:
                all_points.append({
                    "acc": m["accuracy"] * 100,
                    "ece": m["ece"] * 100,
                    "method": data.get("tta_method", data.get("geo_frac", "?")),
                    "clf": data.get("classifier", "?"),
                })

    if not all_points:
        print("  [skip] No results for accuracy vs ECE plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    methods = sorted(set(p["method"] for p in all_points))
    cmap = plt.cm.tab10
    for i, method in enumerate(methods):
        pts = [p for p in all_points if p["method"] == method]
        ax.scatter([p["acc"] for p in pts], [p["ece"] for p in pts],
                   label=str(method), alpha=0.7, color=cmap(i % 10), s=40)

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("ECE (%)")
    ax.set_title("Accuracy vs Calibration Error")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    fig.tight_layout()
    out = output_dir / "accuracy_vs_ece.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [saved] {out}")

def plot_domain_shift_analysis(results_dir: Path, output_dir: Path, args):
    """
    Generates a publication-quality publication grid separating domain shift 
    metrics into categorical groups with error bars.
    """
    # Assuming your data is saved in your results directory
    f = results_dir / "domain_stats" / "_imagenet_and_test_r.json"
    if not f.exists():
        print("  [skip] Domain shift analysis JSON not found")
        return

    data = _load_json(f)
    global_stats = data.get("global_summary", {})
    if not global_stats:
        return

    # Define logical groups for bounded metrics [0, 1] to avoid scaling issues
    metric_groups = {
        "Perceptual & Structure": [
            ("SSIM", "ssim_mean", "ssim_std"),
            #("Luminance SSIM", "luminance_ssim_mean", "luminance_ssim_std"),
            ("LPIPS Score", "lpips_score_mean", "lpips_score_std"),
        ],
        "Color & Distribution": [
            ("Histogram Dist.", "histogramm_distance_mean", "histogramm_distance_std"),
            ("Color Moment Dist.", "color_moment_distance_mean", "color_moment_distance_std"),
            #("Wasserstein Dist.", "wasserstein_distance_mean", "wasserstein_distance_std"),
        ],
        "Edges & Boundaries": [
            ("Edge Similarity", "edge_similarity_mean", "edge_similarity_std"),
            #("HED Distance", "hed_dists_mean", "hed_dists_std"),
            ("LDC Distance", "ldc_dists_mean", "ldc_dists_std"),
            #("HED Figure of Merit", "hed_fom_mean", "hed_fom_std"),
            ("LDC Figure of Merit", "ldc_fom_mean", "ldc_fom_std"),
        ],
        "Depth & Geometry": [
            ("DepthAnything MAE", "depthanything_v2_large_mae_mean", "depthanything_v2_large_mae_std"),
            ("DPT MAE", "dpt_large_mae_mean", "dpt_large_mae_std"),
            ("DepthAnything Spear.", "depthanything_v2_large_spear_mean", "depthanything_v2_large_spear_std"),
            ("DPT Spearman", "dpt_large_spear_mean", "dpt_large_spear_std"),
        ]
    }

    # Setup 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = plt.get_cmap("Set2").colors

    for idx, (group_name, metrics) in enumerate(metric_groups.items()):
        ax = axes[idx]
        
        labels, means, stds = [], [], []
        for name, mean_k, std_k in metrics:
            if mean_k in global_stats:
                labels.append(name)
                means.append(global_stats[mean_k])
                stds.append(global_stats.get(std_k, 0))

        if not labels:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        y_pos = np.arange(len(labels))
        # Plot horizontal bars with standard deviation error bars
        bars = ax.barh(y_pos, means, xerr=stds, align='center', alpha=0.8, 
                       color=colors[:len(labels)], edgecolor='none', capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()  # Top-down order
        ax.set_title(group_name, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(0, 1.1)  # All these metrics are bounded near/within [0, 1]
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Add data values on top of the bars
        for bar, mean in zip(bars, means):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{mean:.2f}', 
                    ha='left', va='center', fontsize=9, fontweight='semibold')

    fig.suptitle("Domain Shift Metric Analysis Summary", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    
    out = output_dir / "domain_shift_metrics_summary.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [saved] {out}")

try:
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def plot_domain_shift_class_scatter_per_group(results_dir: Path, output_dir: Path, args=None):
    """
    Generates spacious class-wise scatter plots for domain shift metrics.
    Saves a single clean multi-page PDF document (one page per group) 
    AND exports separate standalone files inside a dedicated subfolder pipeline.
    """
    f = results_dir / "domain_stats" / "_imagenet_and_test_r.json"
    if not f.exists():
        print("  [skip] Class domain shift scatter data not found")
        return

    data = _load_json(f)
    class_summaries = data.get("class_summaries", {})
    name_list = _get_names(args.split)
    if not class_summaries:
        print("  [skip] No class-wise summaries available")
        return

    # Define groups for bounded metrics [0, 1] 
    metric_groups = {
        "perceptual_structure": ("Perceptual & Structure Metrics", [
            ("ssim_mean", "SSIM"),
            #("luminance_ssim_mean", "Luminance SSIM"),
            ("lpips_score_mean", "LPIPS Score"),
        ]),
        "color_distribution": ("Color & Distribution Metrics", [
            ("histogramm_distance_mean", "Histogram Dist."),
            ("color_moment_distance_mean", "Color Moment Dist."),
            #("wasserstein_distance_mean", "Wasserstein Dist."),
        ]),
        "edges_boundaries": ("Edges & Boundaries Metrics", [
            ("edge_similarity_mean", "Edge Similarity"),
            #("hed_dists_mean", "HED Distance"),
            ("ldc_dists_mean", "LDC Distance"),
            #("hed_fom_mean", "HED Figure of Merit"),
            ("ldc_fom_mean", "LDC Figure of Merit"),
        ]),
        "depth_geometry": ("Depth & Geometry Metrics", [
            ("depthanything_v2_large_mae_mean", "DepthAnything MAE"),
            ("dpt_large_mae_mean", "DPT MAE"),
            ("depthanything_v2_large_spear_mean", "DepthAnything Spear."),
            ("dpt_large_spear_mean", "DPT Spearman"),
        ])
    }

    # FIX 1: Gather data map structured correctly by metric key string
    classes_data = {}
    for class_id, metrics in class_summaries.items():
        # Get human-readable name for the class ID (e.g. "tench")
        class_name = name_list[int(class_id)]
        for k, val in metrics.items():
            if k.endswith("_mean"):
                # Key on the metric string (e.g., 'ssim_mean'), append tuple of (class_id, value)
                classes_data.setdefault(k, []).append((class_id, val))

    unique_classes = sorted(list(class_summaries.keys()), key=int)
    color_map = plt.cm.get_cmap("tab20", max(len(unique_classes), 2))
    class_colors = {cls: color_map(i) for i, cls in enumerate(unique_classes)}

    # Create the dedicated subfolder directory path for isolated assets
    subfolder_dir = output_dir / "domain_shift_groups"
    subfolder_dir.mkdir(parents=True, exist_ok=True)

    # Master path configuration for the multi-page output target
    multipage_pdf_path = output_dir / "domain_shift_class_scatter_multipage.pdf"
    
    print(f"  [processing] Splitting charts into isolated pages and individual files...")

    # Open multi-page document context manager stream
    with PdfPages(multipage_pdf_path) as pdf:
        for group_id, (group_name, metrics_list) in metric_groups.items():
            # Create a broad, short landscape figure ideal for single metric rows
            fig, ax = plt.subplots(figsize=(10, 4.5))
            
            y_ticks = []
            y_labels = []
            
            for y_idx, (metric_key, display_name) in enumerate(metrics_list):
                if metric_key not in classes_data:
                    continue
                    
                y_ticks.append(y_idx)
                y_labels.append(display_name)
                
                # Plot layout backdrop alignment grid
                ax.axhline(y_idx, color='gray', linestyle=':', alpha=0.3, zorder=1)
                
                # FIX 2: Correctly unpacking class_id (as string) and metric value
                for class_id, val in classes_data[metric_key]:
                    # Seed deterministic class jitter to preserve visualization spacing 
                    np.random.seed(int(class_id) + 42)
                    jitter = np.random.uniform(-0.06, 0.06)
                    
                    # Look up class human readable label for the legend
                    class_name = name_list[int(class_id)]
                    
                    ax.scatter(val, y_idx + jitter, 
                               color=class_colors[class_id], 
                               edgecolor='black', linewidth=0.6,
                               s=85, alpha=0.9, zorder=2,
                               label=class_name if y_idx == 0 else "")

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=11)
            ax.set_ylim(-0.6, len(metrics_list) - 0.4)
            ax.invert_yaxis()
            ax.set_xlim(-0.05, 1.05)
            ax.set_xlabel("Metric Value Space", fontsize=10, labelpad=6)
            ax.set_title(group_name, fontsize=13, fontweight="bold", pad=12)
            ax.grid(axis='x', linestyle='--', alpha=0.5)

            # Localized clean plot area legends
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), 
                          loc='upper right', frameon=True, fontsize=9.5)

            fig.tight_layout()
            
            # Target 1: Commit visualization block as a new separate page in the master PDF document stream
            pdf.savefig(fig, bbox_inches='tight')
            
            # Target 2: Save an explicit standalone isolated figure image inside the directory path
            individual_out = subfolder_dir / f"domain_shift_{group_id}.pdf"
            fig.savefig(individual_out, bbox_inches='tight')
            plt.close(fig)

    print(f"  [saved multi-page] {multipage_pdf_path}")
    print(f"  [saved individual groups] check directory: {subfolder_dir}/")



def plot_metric_correlation_heatmap(results_dir: Path, output_dir: Path, args=None):
    """
    Computes and plots a Pearson correlation heatmap across all domain shift metrics
    to identify statistical redundancy and streamline the thesis narrative.
    """
    if not HAS_SNS:
        print("  [skip] Seaborn not installed. Skipping correlation matrix.")
        return

    f = results_dir / "domain_stats" / "_imagenet_and_test_r.json"
    if not f.exists():
        print("  [skip] Metrics JSON not found for correlation analysis.")
        return

    data = _load_json(f)
    class_summaries = data.get("class_summaries", {})
    if not class_summaries:
        return

    # Map your metrics to clean, readable text abbreviations for the axis labels
    metric_mapping = {
        "ssim_mean": "SSIM",
        #"luminance_ssim_mean": "Lum. SSIM",
        "lpips_score_mean": "LPIPS Score",
        "histogramm_distance_mean": "Hist. Dist",
        "color_moment_distance_mean": "Color Mom. Dist",
        #"wasserstein_distance_mean": "Wasserstein Dist",
        "edge_similarity_mean": "Edge Similarity",
        #"hed_dists_mean": "HED Distance",
        "ldc_dists_mean": "LDC Distance",
        #"hed_fom_mean": "HED FOM",
        "ldc_fom_mean": "LDC FOM",
        "depthanything_v2_large_mae_mean": "DepthAnything MAE",
        "dpt_large_mae_mean": "DPT MAE",
        "depthanything_v2_large_spear_mean": "DepthAnything Spear.",
        "dpt_large_spear_mean": "DPT Spearman",
    }

    # 1. Gather all class mean data into a structured table
    all_rows = {}
    for class_id, metrics in class_summaries.items():
        # Only pull metrics that match our mapped keys and end with _mean
        all_rows[int(class_id)] = {
            metric_mapping[k]: v for k, v in metrics.items() if k in metric_mapping
        }
    
    df = pd.DataFrame.from_dict(all_rows, orient="index")
    
    if df.empty:
        print("  [skip] DataFrame empty for correlation mapping.")
        return

    # 2. Compute the Pearson Correlation Matrix
    corr_matrix = df.corr(method="pearson")

    # 3. Plotting preparation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a mask to hide the upper triangle (since correlation matrices are symmetrical)
    # This prevents visual clutter and makes the chart look highly professional
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",      # Red = Positive, Blue = Negative, White = No correlation
        vmax=1.0, vmin=-1.0,  # Forces boundaries strictly to Pearson r range [-1, 1]
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"label": "Pearson Correlation Coefficient ($r$)", "shrink": 0.8},
        annot=True,           # Prints the exact decimal values inside the boxes
        fmt=".2f",            # Formats to 2 decimal places
        annot_kws={"size": 7, "weight": "semibold"} # Small, clean typography
    )

    # Clean text rotations so they don't overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    ax.set_title("Domain Shift Metrics Redundancy Analysis\n(Correlation Across 200 Shared Classes)", 
                 fontsize=12, fontweight="bold", pad=15)
    
    fig.tight_layout()
    out_file = output_dir / "metric_correlation_matrix.pdf"
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  [saved correlation map] {out_file}")

def plot_domain_shift_class_rankings_by_group(results_dir: Path, output_dir: Path, args=None):
    """
    Generates isolated, standalone PDF files for each metric group saved inside a 
    dedicated subfolder pipeline. Parallel columns list class names ordered from best to worst.
    
    The top 20 classes from the FIRST metric column are tracked across all subsequent 
    metric columns on that page using matching color fills for easy visual scanning.
    """
    f = results_dir / "domain_stats" / "_imagenet_and_test_r.json"
    if not f.exists():
        print("  [skip] Metrics JSON not found for rankings analysis.")
        return

    data = _load_json(f)
    class_summaries = data.get("class_summaries", {})
    if not class_summaries:
        print("  [skip] No class-wise summaries available for rankings.")
        return

    split_str = args.split if args else "test_r"
    try:
        name_list = _get_names(split_str)
    except Exception:
        name_list = [f"Class {i}" for i in range(1000)]

    # Structural analytical groups with explicitly assigned sorting properties
    # Tuple pattern: (metric_key_string, column_display_label, reverse_sort_boolean)
    metric_groups = {
        "perceptual_structure": ("Perceptual & Structure Metrics", [
            ("ssim_mean", "SSIM", False),
            ("lpips_score_mean", "LPIPS Score", True),
        ]),
        "color_distribution": ("Color & Distribution Metrics", [
            ("histogramm_distance_mean", "Histogram Dist.", True),
            ("color_moment_distance_mean", "Color Moment Dist.", True),
        ]),
        "edges_boundaries": ("Edges & Boundaries Metrics", [
            ("edge_similarity_mean", "Edge Similarity", False),
            ("ldc_dists_mean", "LDC Distance", True),
            ("ldc_fom_mean", "LDC Figure of Merit", False),
        ]),
        "depth_geometry": ("Depth & Geometry Metrics", [
            ("depthanything_v2_large_mae_mean", "DepthAnything MAE", True),
            ("dpt_large_mae_mean", "DPT MAE", True),
            ("depthanything_v2_large_spear_mean", "DepthAnything Spear.", False),
            ("dpt_large_spear_mean", "DPT Spearman", False),
        ])
    }

    # Restructure source JSON dictionary records down into accessible flat metrics arrays
    metrics_data = {}
    for class_id, metrics in class_summaries.items():
        try:
            class_name = name_list[int(class_id)]
        except IndexError:
            class_name = f"ID {class_id}"
            
        for k, val in metrics.items():
            if k.endswith("_mean"):
                metrics_data.setdefault(k, []).append((class_name, val))

    # Initialize the target dedicated pipeline subfolder directory
    subfolder_dir = output_dir / "domain_shift_rankings"
    subfolder_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [processing] Exporting distinct group rank summaries to: {subfolder_dir}/")

    # Generate isolated assets processing one group at a time
    for group_id, (group_display_title, metrics_list) in metric_groups.items():
        active_metrics = [m for m in metrics_list if m[0] in metrics_data]
        if not active_metrics:
            continue
            
        n_cols = len(active_metrics)
        column_data = []
        column_headers = []
        
        # Track raw class records to easily run identity matching evaluations later
        # matrix dimensions: [col_idx][row_idx] -> class_name string
        raw_class_matrix = [] 

        for col_idx, (key, display_name, minimize_val) in enumerate(active_metrics):
            # Sort records: lower values on top if minimize_val=True, else higher on top
            sorted_records = sorted(metrics_data[key], key=lambda x: x[1], reverse=not minimize_val)
            
            # Extract raw sorted class arrays for index tracking checks
            raw_class_matrix.append([record[0] for record in sorted_records])
            
            # Construct formatted text cells with directional indicator arrows
            arrow = "↓" if minimize_val else "↑"
            column_headers.append(f"{display_name} {arrow}")
            
            formatted_col = []
            for rank, (name, val) in enumerate(sorted_records, start=1):
                short_name = name[:14] + ".." if len(name) > 16 else name
                formatted_col.append(f"{rank}. {short_name} ({val:.2f})")
            column_data.append(formatted_col)

        # Transpose column vectors into rows for the final table structure
        table_rows = list(zip(*column_data))
        n_rows = len(table_rows)
        if n_rows == 0:
            continue

        # Generate unique, visually distinct colors for tracking the Top 20 items
        # We use a soft pastel palette so text remains perfectly readable without high contrast glare
        cmap = plt.cm.get_cmap("Pastel1", 20)
        top_20_classes_first_col = raw_class_matrix[0][:20]
        class_color_mapping = {class_name: cmap(i) for i, class_name in enumerate(top_20_classes_first_col)}

        # Scale figure canvas dynamic heights using structural row length allocations
        fig_height = max(8, n_rows * 0.24)
        fig, ax = plt.subplots(figsize=(3.4 * n_cols, fig_height))
        ax.axis('off')

        col_widths = [1.0 / n_cols] * n_cols
        table = ax.table(
            cellText=table_rows,
            colLabels=column_headers,
            colWidths=col_widths,
            loc='center',
            cellLoc='left'
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)

        # Apply specific visual styles to headers and match classes to highlight fills
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_text_props(weight='bold', color='white', size=10)
                cell.set_facecolor('#2c3e50')  # Dark slate gray professional header
                cell.set_height(0.035)
            else:
                cell.set_height(max(0.012, 1.0 / (n_rows + 5)))
                cell.set_linewidth(0.3)
                
                # Retrieve the identity string of the class assigned to this cell
                current_cell_class = raw_class_matrix[col_idx][row_idx - 1]
                
                # Check if this class is one of the original top 20 from column 0
                if current_cell_class in class_color_mapping:
                    # Paint cell with its distinct tracked tracking color background
                    cell.set_facecolor(class_color_mapping[current_cell_class])
                    # Add bold text treatment so tracked entities jump out across columns
                    cell.set_text_props(weight='bold')
                else:
                    # Non-tracked standard clean baseline alternate row tints
                    if row_idx % 2 == 0:
                        cell.set_facecolor('#f8f9fa')

        fig.suptitle(f"{group_display_title}\n(Top 20 of First Metric Color-Tracked Across Columns)", 
                     fontsize=12, fontweight="bold", y=0.99)
        
        fig.tight_layout()
        group_out_path = subfolder_dir / f"rankings_{group_id}.pdf"
        fig.savefig(group_out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"    [saved individual group rank] {group_out_path.name}")

    print(f"  [completed] Check folder pipeline target at: {subfolder_dir}/")
#  =========================================================================
# Main
# =========================================================================
def generate_all_plots(args):
    if not HAS_MPL:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib seaborn")
        return

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Result Visualisation")
    print("=" * 60)

    #plot_style_transfer_comparison(results_dir, output_dir, args)
    #plot_ablation_bars(results_dir, output_dir, "retrieval", args)
    #plot_ablation_bars(results_dir, output_dir, "eval", args)
    #plot_ablation_bars(results_dir, output_dir, "nrefs", args)
    #plot_nrefs_sweep(results_dir, output_dir, args)
    #plot_hybrid_tta(results_dir, output_dir, args)
    #plot_accuracy_vs_ece(results_dir, output_dir, args)
    #plot_domain_shift_analysis(results_dir, output_dir, args)
    #plot_domain_shift_class_scatter_per_group(results_dir, output_dir, args)
    plot_domain_shift_class_rankings_by_group(results_dir, output_dir, args)

    print(f"\nAll figures written to {output_dir}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate thesis figures")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--output_dir", type=str, default="./figures")
    p.add_argument("--dataset", type=str, default="imagenet")
    p.add_argument("--split", type=str, default="test_r")
    return p


def main():
    args = build_parser().parse_args()
    generate_all_plots(args)


if __name__ == "__main__":
    main()
