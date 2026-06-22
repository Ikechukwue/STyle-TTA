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
from config.constants import ALL_CLASSIFIERS
from config.helpers import calc_top_k
from .helpers.support_funct import *
from .helpers.ablation import plot_ablation_bars, plot_nrefs_sweep, plot_topk_confidence
from .helpers.style_transfer import plot_hybrid_tta, plot_style_transfer_comparison
from .helpers.domain_difference import (plot_domain_shift_analysis, plot_domain_shift_class_scatter_per_group,
                                        plot_domain_shift_class_rankings_by_group,
                                        ) 




# =========================================================================
# Accuracy vs ECE Scatter
# =========================================================================
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

def plot_classifier_bases(
    results_dir: str, 
    dataset="imagenet", 
    splits=["test_r", "val@test_r", "val"]
):
    """
    Parses raw prediction json files from a results directory, groups models 
    by paradigm, and generates two publication-quality grouped bar charts:
    one for Top-1 Accuracy and one for Top-5 Accuracy.
    """
    group_mapping = {
        "resnet18": "CNN",
        "densenet121": "CNN",
        "swin_base_patch4_window7_224": "ViT",
        "vit_base_patch16_224": "ViT",
        "ViT-B-16": "VLM",
        "dinov2_vitb14": "FM"
    }

    results_path = Path(results_dir)
    model_data = {}
    
    for s in splits: 
        for cla in ALL_CLASSIFIERS:
            full_preds_path = None
            preds_path = f"{cla}_geometric_vanilla_nviews1_seed71397589.json"

            if s == "test_r":
                full_preds_path = results_path / "baseline/tta_inference/predictions/imagenet/test_r" / preds_path
            elif s == "val@test_r":
                full_preds_path = results_path / "geometric_tta/tta_inference/predictions/imagenet/val@test_r" / preds_path
            
            if full_preds_path is None or not full_preds_path.exists():
                continue
                
            try:
                data = load_json(str(full_preds_path))
                config = data.get("config", {})

                top1_acc = calc_top_k(full_preds_path, k=1)
                top5_acc = calc_top_k(full_preds_path, k=5)
                
                if top1_acc is None or top5_acc is None:
                    continue

                model_name = cla
                if model_name not in model_data:
                    model_data[model_name] = {
                        "group": group_mapping[model_name],
                        "top1": {sp: 0.0 for sp in splits},
                        "top5": {sp: 0.0 for sp in splits}
                    }
                
                split = config.get("split", s)
                model_data[model_name]["top1"][split] = top1_acc
                model_data[model_name]["top5"][split] = top5_acc
                
            except Exception as e:
                print(f"[Warning] Failed to process {full_preds_path}: {e}")
                continue

    if not model_data:
        print("[Error] No valid prediction data found matching criteria.")
        return

    sorted_models = sorted(model_data.items(), key=lambda x: x[1]["group"])
    
    models = [m[0] for m in sorted_models]
    groups = [m[1]["group"] for m in sorted_models]
    x_labels = [f"{m}\n({g})" for m, g in zip(models, groups)]
    
    def generate_plot(metric_type: str, title_label: str, filename: str):
        x = np.arange(len(models))
        width = 0.25
        n_splits = len(splits)
        start_offset = -(n_splits - 1) * width / 2
        
        fig, ax = plt.subplots(figsize=(13, 6.5))
        colors = ['#1f77b4', '#aec7e8', '#ff7f0e'] 
        
        for i, split in enumerate(splits):
            offset = start_offset + i * width
            scores = [model_data[m][metric_type][split] for m in models]
            ax.bar(x + offset, scores, width, label=split, color=colors[i % len(colors)], edgecolor='black', alpha=0.85)
            
        ax.set_ylabel(f'{title_label} Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Classifiers (Grouped by Paradigm)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title(f'Classifier Robustness Comparison - {title_label} ({dataset.upper()})', fontsize=14, fontweight='bold', pad=18)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15, ha='right', fontsize=10)
        
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(title="Dataset Splits", fontsize=10, title_fontsize=11, loc='upper left', frameon=True)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Successfully saved thesis figure: {filename}")

    generate_plot("top1", "Top-1", f"{dataset}_classifier_comparison_top1.png")
    generate_plot("top5", "Top-5", f"{dataset}_classifier_comparison_top5.png")

def plot_accuracy_vs_ece(results_dir: Path, output_dir: Path, args):
    """Publication-quality scatter plot of accuracy vs ECE across all experiments."""
    all_points = []
    # (Your existing search and load logic remains identical)
    for pattern in ["thesis/geometric_tta", "thesis/ablation", "thesis/hybrid_tta",
                    "geometric_tta", "ablation/retristyle","ablation/adain_tta", "hybrid_tta"]:
        d = results_dir / pattern / f"tta_inference/results/{args.dataset}/{args.split}"
        for f in find_json(d, "*.json"):
            data = load_json(f)
            m = data.get("metrics", {})
            if "accuracy" in m and "ece" in m:
                # Multiply by 100 if your data is in [0, 1] range
                acc_val = m["accuracy"] * 100 if m["accuracy"] <= 1.0 else m["accuracy"]
                ece_val = m["ece"] * 100 if m["ece"] <= 1.0 else m["ece"]
                
                all_points.append({
                    "acc": acc_val,
                    "ece": ece_val,
                    "method": data.get("tta_method", "?"),
                    "strategy": data.get("eval_strategy", "zero"),  # Crucial to capture vanilla vs zero!
                    "clf": data.get("classifier", "?"),
                })

    if not all_points:
        print("  [skip] No results for accuracy vs ECE plot")
        return

    # --- STYLE & CONFIGURATION SETTINGS ---
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    
    # 1. Map Classifiers to unique, distinct markers
    clf_markers = {
        "resnet18": "o",       # Circle
        "densenet121": "s",    # Square
        "vit_base_patch16_224": "^", # Triangle Up
        "swin_base_patch4_window7_224": "D", # Diamond
        "dinov2_vitb14": "p",  # Pentagon
        "?": "X"
    }
    
    # 2. Build explicit color palette for methods + strategy combinations
    # This highlights why vanilla/tpt are better than zero!
    unique_methods = sorted(set(p["method"] for p in all_points))
    cmap = plt.cm.get_cmap("tab10")
    color_map = {method: cmap(i % 10) for i, method in enumerate(unique_methods)}

    # Track what we've added to the legend to avoid massive duplicate lists
    legend_tracker = {}

    # --- PLOTTING DATA POINTS ---
    for p in all_points:
        # Style logic: Give 'vanilla' and 'tpt' a clean filled look, and 'zero' a slightly translucent look
        alpha_val = 0.9 if p["strategy"] in ["vanilla", "tpt"] else 0.4
        edge_color = "black" if p["strategy"] in ["vanilla", "tpt"] else "none"
        line_width = 0.8 if p["strategy"] in ["vanilla", "tpt"] else 0
        
        marker = clf_markers.get(p["clf"], "X")
        color = color_map.get(p["method"], "#7f7f7f")
        
        # Label handling for grouped legend
        lbl = f"{p['method']} ({p['strategy']})"
        label_key = (lbl, p["clf"])
        
        scatter_handle = ax.scatter(
            p["acc"], p["ece"],
            color=color,
            marker=marker,
            s=80,  # Larger size for publication visibility
            alpha=alpha_val,
            edgecolors=edge_color,
            linewidths=line_width
        )
        
    # --- VISUAL ENHANCEMENTS FOR THE THESIS ---
    # Draw the "Ideal Zone" boundary box in the bottom right corner
    max_acc = max([p["acc"] for p in all_points]) if all_points else 100
    min_ece = min([p["ece"] for p in all_points]) if all_points else 0
    
    ax.axhspan(0, 10, xmin=0.6, xmax=1.0, color='green', alpha=0.05, label='Optimal Performance Zone')
    
    # Clean labels & Grid adjustments
    ax.set_xlabel("Top-1 Accuracy (%)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Expected Calibration Error / ECE (%)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title("Empirical Trade-off: Accuracy vs. Calibration Calibration", fontsize=13, fontweight='bold', pad=15)
    
    # Custom Legend Reconstruction (Split into Methods vs. Backbones)
    from matplotlib.lines import Line2D
    
    # Method elements (Colors)
    color_legends = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[m], markersize=10, label=m) for m in unique_methods]
    # Backbone elements (Shapes)
    unique_clfs = sorted(set(p["clf"] for p in all_points))
    shape_legends = [Line2D([0], [0], marker=clf_markers.get(c, "X"), color='w', markerfacecolor='gray', markersize=10, label=c) for c in unique_clfs]
    
    first_legend = ax.legend(handles=color_legends, title="TTA Algorithm", loc="upper right", frameon=True, fontsize=9)
    ax.add_artist(first_legend)
    ax.legend(handles=shape_legends, title="Classifier Backbone", loc="lower left", frameon=True, fontsize=9)

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    
    out = output_dir / "accuracy_vs_ece_upgraded.pdf"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved upgraded plot] {out}")
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
    #plot_style_transfer_comparison(results_dir, output_dir, args)
    #plot_accuracy_vs_ece(results_dir, output_dir, args)
    metric_path = "/home/stud/nemmler/retristyle/results/classifier_eval/classifier_evaluation/metrics_base.json" 
    plot_classifier_bases("./results")
    anchors = {
            "best_retrieval": "dino",  # The strategy held constant for Eval/nrefs plots
            "best_eval": "zero",       # The strategy held constant for Retrieval/nrefs plots
        }
    
    for method in []: #['ablation/adain', 'ablation/retristyle', 'geometric_tta']: #, 
        plot_topk_confidence(results_dir, output_dir, args, method, k=args.top_k)
        if args.top_k != 1:
            plot_topk_confidence(results_dir, output_dir, args, method, k=1)
        anchors["best_n_refs"] = 16 if method == 'ablation/retristyle' else 32
        plot_ablation_bars(results_dir, output_dir, "retrieval", method, args, **anchors)
        plot_ablation_bars(results_dir, output_dir, "eval",method,  args, **anchors)
        plot_ablation_bars(results_dir, output_dir, "nrefs",method,  args, **anchors)
        plot_nrefs_sweep(results_dir, output_dir, args, method,
                            best_retrieval=anchors["best_retrieval"], 
                            best_eval=anchors["best_eval"])
        
    #plot_hybrid_tta(results_dir, output_dir, args)
    #plot_accuracy_vs_ece(results_dir, output_dir, args)
    
    
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
    p.add_argument("--top_k", type=int, default=5)
    return p


def main():
    args = build_parser().parse_args()
    generate_all_plots(args)


if __name__ == "__main__":
    main()
