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
from typing import Dict, List, Optional, Tuple

import numpy as np

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


# =========================================================================
# 1. Style Transfer Method Comparison (Grouped Bar)
# =========================================================================
def plot_style_transfer_comparison(results_dir: Path, output_dir: Path):
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
def plot_ablation_bars(results_dir: Path, output_dir: Path, ablation_type: str):
    """Bar chart for a specific ablation axis."""
    abl_dir = results_dir / "thesis" / "ablation"
    if not abl_dir.exists():
        abl_dir = results_dir / "ablation"
    files = _find_json(abl_dir, "*_results.json")
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
def plot_nrefs_sweep(results_dir: Path, output_dir: Path):
    """Line plot of accuracy vs n_refs per classifier."""
    abl_dir = results_dir / "thesis" / "ablation"
    if not abl_dir.exists():
        abl_dir = results_dir / "ablation"
    files = _find_json(abl_dir, "*_results.json")
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
def plot_hybrid_tta(results_dir: Path, output_dir: Path):
    """Line plot of accuracy across geo/style mixing ratios."""
    hybrid_dir = results_dir / "thesis" / "hybrid_tta"
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
def plot_accuracy_vs_ece(results_dir: Path, output_dir: Path):
    """Scatter plot of accuracy vs ECE across all experiments."""
    all_points = []
    for pattern in ["thesis/geometric_tta", "thesis/ablation", "thesis/hybrid_tta",
                    "geometric_tta", "ablation", "hybrid_tta"]:
        d = results_dir / pattern
        for f in _find_json(d, "*_results.json"):
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


# =========================================================================
# Main
# =========================================================================
def generate_all_plots(results_dir: Path, output_dir: Path):
    if not HAS_MPL:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib seaborn")
        return

    _setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Result Visualisation")
    print("=" * 60)

    plot_style_transfer_comparison(results_dir, output_dir)
    plot_ablation_bars(results_dir, output_dir, "retrieval")
    plot_ablation_bars(results_dir, output_dir, "eval")
    plot_ablation_bars(results_dir, output_dir, "nrefs")
    plot_nrefs_sweep(results_dir, output_dir)
    plot_hybrid_tta(results_dir, output_dir)
    plot_accuracy_vs_ece(results_dir, output_dir)

    print(f"\nAll figures written to {output_dir}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate thesis figures")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--output_dir", type=str, default="./figures")
    return p


def main():
    args = build_parser().parse_args()
    generate_all_plots(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
