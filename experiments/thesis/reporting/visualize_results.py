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
from config.constants import ALL_CLASSIFIERS, TTA_STRATEGIES, ALL_SEEDS, VIT_CLASSIFIERS, VLM_CLASSIFIERS, CNN_CLASSIFIERS, FM_CLASSIFIERS
from config.helpers import calc_top_k
from .helpers.support_funct import *
from .helpers.ablation import plot_ablation_bars, plot_nrefs_sweep, plot_all_topk_metrics,plot_ablation_nrefs_multi, plot_ablation_lines, plot_ablation_nrefs, plot_all_ablation_nrefs, plot_all_ablation_retr
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

def plot_accuracy_vs_ece_integrated(results_dir: Path, output_dir: Path, strategy_keys: list, args):
    from matplotlib.lines import Line2D
    import numpy as np
    import matplotlib.pyplot as plt

    # Group FM and VLM together
    clf_groups = {
        "CNN": CNN_CLASSIFIERS,
        "ViT": VIT_CLASSIFIERS,
        "VLM/FM": VLM_CLASSIFIERS + FM_CLASSIFIERS
    }
    group_markers = {"CNN": "o", "ViT": "s", "VLM/FM": "D"}
    
    aggregated = {s: {g: {} for g in clf_groups} for s in strategy_keys}
    
    for s_key in strategy_keys:
        cfg = TTA_STRATEGIES[s_key]
        for g_name, cl_list in clf_groups.items():
            for cl in cl_list:
                for rfs in cfg["axis"]:
                    for seed in ALL_SEEDS:
                        f_name = cfg["template"].format(cl=cl, rfs=rfs, seed=seed)
                        f = results_dir / s_key / f"tta_inference/results/{args.dataset}/{args.split}" / f_name
                        if not f.exists(): continue
                        
                        data = load_json(f)
                        m = data.get("metrics", {})
                        if "accuracy" in m and "ece" in m:
                            if rfs not in aggregated[s_key][g_name]:
                                aggregated[s_key][g_name][rfs] = {"acc": [], "ece": []}
                            aggregated[s_key][g_name][rfs]["acc"].append(m["accuracy"] * 100)
                            aggregated[s_key][g_name][rfs]["ece"].append(m["ece"] * 100)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    for s_key in strategy_keys:
        cfg = TTA_STRATEGIES[s_key]
        for g_name, rfs_data in aggregated[s_key].items():
            for rfs, vals in rfs_data.items():
                if not vals["acc"]: continue
                
                avg_acc = np.mean(vals["acc"])
                avg_ece = np.mean(vals["ece"])
                
                shade_idx = list(cfg["axis"]).index(rfs) / max(1, len(cfg["axis"]) - 1)
                color = plt.cm.get_cmap(cfg["color_shade"])(0.3 + 0.6 * shade_idx)
                
                ax.scatter(avg_acc, avg_ece, color=color, marker=group_markers.get(g_name, "x"), 
                           s=100, zorder=3)

    legend_elements = []
    # Strategy section
    for s in strategy_keys:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=plt.cm.get_cmap(TTA_STRATEGIES[s]["color_shade"])(0.6), 
                                      markersize=8, label=f"Method: {TTA_STRATEGIES[s]['label']}"))
    
    legend_elements.append(Line2D([0], [0], color='w')) 
    
    # Backbone section
    for g, marker in group_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                      markerfacecolor='gray', markersize=8, label=f"Backbone: {g}"))

    # Add Note to Legend Box
    legend_elements.append(Line2D([0], [0], color='w', label="Note: Darker color \nintensity = higher N-Refs"))

    ax.legend(handles=legend_elements, loc="best", frameon=True, fontsize=9, title="Plot Legend")

    ax.set_xlabel("Average Top-1 Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Average ECE (%)", fontsize=11, fontweight='bold')
    ax.set_title(f"Accuracy vs. ECE: ImageNet-R", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(output_dir / "avg_accuracy_vs_ece.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

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
        "best_retrieval": "dino",
        "best_eval": "zero",
    }
    #plot_all_topk_metrics(results_dir, args.dataset, args.split)
    #plot_all_ablation_retr(results_dir, output_dir,'ablation/adain_tta',  args)
    done = []
    for method in ['hybrid_tta', 'geometric_tta' ,'ablation/adain_tta',  'ablation/retristyle', ]:
        done.append(method)
        plot_ablation_nrefs_multi(results_dir, output_dir, done, args)
        anchors["best_n_refs"] = 16 #if method == 'ablation/retristyle' else 32
        #plot_all_ablation_nrefs(results_dir, output_dir, method, args, True)
        #plot_ablation_lines(results_dir, output_dir, "retrieval", method, args, **anchors)
        #plot_ablation_lines(results_dir, output_dir, "eval", method, args, **anchors)
        #plot_ablation_lines(results_dir, output_dir, "nrefs", method, args, **anchors)
        #plot_nrefs_sweep(results_dir, output_dir, args, method,
        #                 best_retrieval=anchors["best_retrieval"], 
        #                 best_eval=anchors["best_eval"])
        
    # --- New Best of Best Comparison Run ---
    print("\nGenerating Best-of-Best Method Comparison Profiles...")
    #plot_accuracy_vs_ece_integrated(results_dir, output_dir, done, args)

    #plot_method_comparison(results_dir, output_dir, args, best_settings)

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
