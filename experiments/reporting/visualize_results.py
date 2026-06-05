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
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HAS_MPL = True

from .helpers.support_funct import *
from .helpers.ablation import plot_ablation_bars, plot_nrefs_sweep
from .helpers.style_transfer import plot_hybrid_tta, plot_style_transfer_comparison
from .helpers.domain_difference import (plot_domain_shift_analysis, plot_domain_shift_class_scatter_per_group,
                                        plot_domain_shift_class_rankings_by_group,
                                        ) 




# =========================================================================
# Accuracy vs ECE Scatter
# =========================================================================
def plot_accuracy_vs_ece(results_dir: Path, output_dir: Path, args):
    """Scatter plot of accuracy vs ECE across all experiments."""
    all_points = []
    for pattern in ["thesis/geometric_tta", "thesis/ablation", "thesis/hybrid_tta",
                    "geometric_tta", "ablation", "hybrid_tta"]:
        d = results_dir / pattern / f"tta_inference/results/{args.dataset}/{args.split}"
        d = results_dir / pattern / f"tta_inference/results/{args.dataset}/{args.split}"
        for f in find_json(d, "*.json"):
            data = load_json(f)
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

#  =========================================================================
# Main
# =========================================================================
def normalize_data(args):

    results_dir = Path(args.results_dir)
    test_data_path = results_dir / "domain_stats" / f"imagenet_{args.split}" / f"0_30_imagenet_{args.split}.json"
    base_data_path = results_dir / "domain_stats" / "imagenet_val" / f"0_30_imagenet_val_{args.split}.json"
    if not test_data_path.exists() or not base_data_path.exists():
        print("[skip] One or both dataset JSON files are missing.")
        return
    raw_data_a = load_json(test_data_path)
    raw_data_b = load_json(base_data_path)

    norm_data_a, norm_data_b = normalize_values_inter(raw_data_a, raw_data_b)

    return norm_data_a, norm_data_b

def generate_all_plots(args):
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Result Visualisation")
    print("=" * 60)

    anchors = {
            "best_retrieval": "dino",  # The strategy held constant for Eval/nrefs plots
            "best_eval": "zero",       # The strategy held constant for Retrieval/nrefs plots
            "best_n_refs": 16          # The count held constant for Retrieval/Eval plots
        }
    
    plot_style_transfer_comparison(results_dir, output_dir, args)
    for method in ['geometric', 'ablation']:
        plot_ablation_bars(results_dir, output_dir, "retrieval", method, args, **anchors)
        plot_ablation_bars(results_dir, output_dir, "eval",method,  args, **anchors)
        plot_ablation_bars(results_dir, output_dir, "nrefs",method,  args, **anchors)
        plot_nrefs_sweep(results_dir, output_dir, args, method,
                            best_retrieval=anchors["best_retrieval"], 
                            best_eval=anchors["best_eval"])
    #plot_hybrid_tta(results_dir, output_dir, args)
    plot_accuracy_vs_ece(results_dir, output_dir, args)
    plot_accuracy_vs_ece(results_dir, output_dir, args)

    #data_a, data_baseline = normalize_data(args)
    #plot_domain_shift_analysis(data_a, data_baseline, output_dir, args)
    #plot_domain_shift_class_scatter_per_group(data_a, data_baseline, output_dir, args)
    #data_a, data_baseline = normalize_data(args)
    #plot_domain_shift_analysis(data_a, data_baseline, output_dir, args)
    #plot_domain_shift_class_scatter_per_group(data_a, data_baseline, output_dir, args)
    #plot_domain_shift_class_rankings_by_group(data, output_dir, args)

    print("\nRunning Stratified Domain Correlation Analysis...")
    #plot_metric_correlation_heatmap(results_dir, output_dir, args)
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
