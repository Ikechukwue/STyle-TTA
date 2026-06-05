from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np 
import matplotlib.pyplot as plt
from .support_funct import load_json, find_json

def plot_hybrid_tta(results_dir: Path, output_dir: Path, args):
    """Line plot of accuracy across geo/style mixing ratios."""
    hybrid_dir = results_dir / "hybrid_tta"
    if not hybrid_dir.exists():
        hybrid_dir = results_dir / "hybrid_tta"
    files = find_json(hybrid_dir, "*_results.json")
    if not files:
        print("  [skip] No hybrid TTA results")
        return

    grouped: Dict[Tuple[float, str], List[float]] = {}
    for f in files:
        data = load_json(f)
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


def plot_style_transfer_comparison(results_dir: Path, output_dir: Path, args):
    """Grouped bar chart comparing style transfer methods."""
    f = results_dir / "style_transfer_eval" / "all_methods_comparison.json"
    if not f.exists():
        print("  [skip] Style transfer comparison not found")
        return

    data = load_json(f)
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

