import os
import re
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from typing import Optional, Dict, Any, Tuple
from experiments.thesis.reporting.helpers.support_funct import load_json, get_names
from experiments.thesis.reporting.pixel_dashboard import load_pipeline_dataset, process_summaries, extract_class_metrics, TTAVisualizer

# Theme Constants
DARK_BG = "#12121f"
PANEL_BG = "#1e1e2e"
SPINE_COL = "#444466"
COL_POS = "#4c9be8"   # Improved
COL_NEG = "#e05c5c"   # Hurt
COL_MEAN = "#f5c842"  # Metric Average

def load_feature_domain_features(feature_domain_path: str, backbone: str) -> pd.DataFrame:
    """Loads per-class MMD/Wasserstein/KL from feature-space domain gap JSON."""
    data = load_json(feature_domain_path)
    per_class = data["domain_gap"]["per_class"]
    rows = []
    for cls_id, metrics in per_class.items():
        rows.append({
            "class_id":   int(cls_id),
            "mmd":        metrics.get("mmd"),
            "wasserstein": metrics.get("wasserstein"),
            "kl_symmetric": metrics.get("kl_symmetric"),
        })
    df = pd.DataFrame(rows)
    df["backbone"] = backbone
    print(f"  Feature-space domain features loaded: {len(df)} classes [{backbone}]")
    return df


FEATURE_DOMAIN_METRICS = {
    "mmd":          "MMD\n(↑ more gap)",
    "wasserstein":  "Wasserstein W2\n(↑ more gap)",
    "kl_symmetric": "KL Divergence\n(↑ more gap)",
}


def analyse_correlation(
    baseline_path: str,
    tta_dir: str,
    val_pred_path: str,
    feature_domain_path: str,
    backbone: str,
    classifier: str,
    output_dir: str,
):
    """
    Correlates feature-space domain gap with TTA utility and domain-induced accuracy drop.

    Signals computed per class:
      - drop[cls]      = acc_val[cls] - acc_test_r_base[cls]   (domain shift hurt)
      - delta_acc[cls] = acc_tta[cls] - acc_test_r_base[cls]   (TTA recovery)

    Correlations run:
      - MMD/W2/KL  vs  drop       → does gap predict sensitivity to domain shift?
      - MMD/W2/KL  vs  delta_acc  → does gap predict TTA utility?
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load TTA results (gives baseline test_r acc + delta_acc) ───────
    print("Loading TTA predictions...")
    df_tta, _ = load_pipeline_dataset(baseline_path, tta_dir, classifier, "retristyle")
    config_summary, _, best_info = process_summaries(df_tta)
    df_best = df_tta[
        (df_tta["nviews"] == best_info["nviews"]) &
        (df_tta["eval_strategy"] == best_info["strategy"])
    ].copy()
    # df_best now has: class_id, base_acc (test_r no-TTA), delta_acc, tta_acc

    # ── 2. Load val@test_r predictions → acc_val per class ───────────────
    print("Loading val@test_r predictions...")
    val_metrics = extract_class_metrics(load_json(val_pred_path))
    val_metrics = val_metrics.rename(columns={
        "acc": "val_acc",
        "conf": "val_conf",
        "ent": "val_ent",
    })

    # ── 3. Merge and compute drop ─────────────────────────────────────────
    df = df_best.merge(val_metrics[["class_id", "val_acc"]], on="class_id", how="inner")
    # drop: how much domain shift hurt (positive = hurt)
    df["drop"] = df["val_acc"] - df["base_acc"]
    print(f"  Classes after merge: {len(df)}")
    print(f"  Mean drop:      {df['drop'].mean():+.4f}  (val_acc - base_test_r_acc)")
    print(f"  Mean delta_acc: {df['delta_acc'].mean():+.4f}  (tta_acc - base_test_r_acc)")

    # ── 4. Load feature-space domain gap ─────────────────────────────────
    domain_df = load_feature_domain_features(feature_domain_path, backbone)
    df = df.merge(domain_df, on="class_id", how="inner")
    print(f"  Classes after domain merge: {len(df)}")
    sim_df = simulate_adaptive_gated_inference(
        df_merged=df, 
        gating_feature="mmd",
        baseline_time_per_img=0.04,     # Adjust based on your system
        retristyle_time_per_img=8.92    # 142.76s / 16 views or total images
    )
    sim_df.to_csv(os.path.join(output_dir, f"{backbone}_gating_simulation.csv"), index=False)
    print(f"  Gating simulation saved. Max compute savings: {sim_df['compute_saved_pct'].max():.2f}%")
    # ── 5. Correlations ───────────────────────────────────────────────────
    targets = {
        "delta_acc": "Δ Accuracy (TTA utility)",
        "drop":      "Accuracy drop (domain sensitivity)",
    }
    gap_metrics = list(FEATURE_DOMAIN_METRICS.keys())

    corr_rows = []
    for target_col, target_label in targets.items():
        for metric in gap_metrics:
            valid = df[[metric, target_col]].dropna()
            rho, p = spearmanr(valid[metric], valid[target_col])
            corr_rows.append({
                "target":        target_col,
                "target_label":  target_label,
                "gap_metric":    metric,
                "rho":           rho,
                "p_value":       p,
                "n":             len(valid),
                "significant":   p < 0.05,
            })

    df_corr = pd.DataFrame(corr_rows)

    print("\n── Feature-space gap correlations ───────────────────────────────────")
    print(df_corr.to_string(index=False, float_format="{:.4f}".format))

    # ── 6. Stratified: helped vs hurt by TTA ─────────────────────────────
    df_helped = df[df["delta_acc"] > 0].copy()
    df_hurt   = df[df["delta_acc"] < 0].copy()

    strat_rows = []
    for group_label, df_grp in [("All", df), ("Helped", df_helped), ("Hurt", df_hurt)]:
        for metric in gap_metrics:
            for target_col in targets:
                valid = df_grp[[metric, target_col]].dropna()
                if len(valid) < 3:
                    continue
                rho, p = spearmanr(valid[metric], valid[target_col])
                strat_rows.append({
                    "group":      group_label,
                    "gap_metric": metric,
                    "target":     target_col,
                    "rho":        rho,
                    "p_value":    p,
                    "n":          len(valid),
                })

    df_strat = pd.DataFrame(strat_rows)

    # ── 7. Save CSVs ──────────────────────────────────────────────────────
    df.to_csv(os.path.join(output_dir, f"{backbone}_merged_class_data.csv"), index=False)
    df_corr.to_csv(os.path.join(output_dir, f"{backbone}_correlations.csv"), index=False)
    df_strat.to_csv(os.path.join(output_dir, f"{backbone}_correlations_stratified.csv"), index=False)

    # ── 8. Plots ──────────────────────────────────────────────────────────
    _plot_scatter_grid(df, df_helped, df_hurt, gap_metrics, backbone, best_info, output_dir)
    _plot_correlation_heatmap(df_strat, gap_metrics, backbone, best_info, output_dir)
    _plot_damage_zone_analysis(df, backbone, output_dir)
    # NEW: Call a function to plot the trade-off curve
    _plot_gating_tradeoff(sim_df, backbone, output_dir)
    return df, df_corr, df_strat

def compute_class_geometry_features(
    val_features: np.ndarray, 
    val_labels: np.ndarray
) -> pd.DataFrame:
    """
    Computes class-level structural geometry from the latent embedding space.
    
    Args:
        val_features: Array of shape (N, feature_dim) containing clean validation embeddings.
        val_labels: Array of shape (N,) containing class indices.
        
    Returns:
        DataFrame containing per-class semantic compactness and boundary proximity.
    """
    unique_classes = np.unique(val_labels)
    centroids = {}
    compactness = {}
    
    # 1. Calculate Centroids and Intra-Class Compactness
    for cls in unique_classes:
        cls_mask = (val_labels == cls)
        cls_feats = val_features[cls_mask]
        
        # Mean embedding represents the class semantic anchor
        centroid = np.mean(cls_feats, axis=0)
        centroids[cls] = centroid
        
        # Compactness: Average cosine distance of instances to their own centroid
        # High value = Sprawling, loose semantic representation
        norm_feats = cls_feats / np.linalg.norm(cls_feats, axis=1, keepdims=True)
        norm_centroid = centroid / np.linalg.norm(centroid)
        cosine_dist = 1.0 - np.dot(norm_feats, norm_centroid)
        compactness[cls] = np.mean(cosine_dist)
        
    # 2. Calculate Boundary Proximity (Distance to Nearest Rival Class)
    boundary_proximity = {}
    cls_list = list(centroids.keys())
    
    for i, cls_a in enumerate(cls_list):
        min_dist = float('inf')
        norm_a = centroids[cls_a] / np.linalg.norm(centroids[cls_a])
        
        for cls_b in cls_list:
            if cls_a == cls_b:
                continue
            norm_b = centroids[cls_b] / np.linalg.norm(centroids[cls_b])
            # Cosine distance between cluster centers
            dist = 1.0 - np.dot(norm_a, norm_b)
            if dist < min_dist:
                min_dist = dist
                
        # Low value = Class is squeezed right against a competing semantic boundary
        boundary_proximity[cls_a] = min_dist

    # Compile into structurally scannable dataframe
    rows = [{
        "class_id": int(cls),
        "intra_class_variance": compactness[cls],
        "boundary_proximity": boundary_proximity[cls]
    } for cls in unique_classes]
    
    return pd.DataFrame(rows)
def simulate_adaptive_gated_inference(
    df_merged: pd.DataFrame, 
    gating_feature: str, 
    baseline_time_per_img: float = 0.04,  # ResNet18 forward pass (approx)
    retristyle_time_per_img: float = 9.0  # 142s total divided by typical subset size
) -> pd.DataFrame:
    """
    Simulates routing inferences dynamically based on a feature space metric threshold.
    
    Args:
        df_merged: DataFrame containing class_id, base_acc, tta_acc, and your gating_feature
        gating_feature: Column name to gate on (e.g., 'mmd', 'intra_class_variance')
    """
    # Sort classes by their vulnerability/gap metric
    df_sorted = df_merged.sort_values(by=gating_feature, ascending=False).copy()
    total_classes = len(df_sorted)
    
    simulation_results = []
    
    # Sweep across thresholds: from applying TTA to ALL classes down to NO classes
    for cut_idx in range(total_classes + 1):
        # Classes above the cut-off index get expensive RetriStyle TTA
        tta_classes = df_sorted.iloc[:cut_idx]["class_id"].values
        
        # Calculate dynamic accuracy across the subset
        df_sorted["dynamic_acc"] = np.where(
            df_sorted["class_id"].isin(tta_classes), 
            df_sorted["tta_acc"], 
            df_sorted["base_acc"]
        )
        composite_accuracy = df_sorted["dynamic_acc"].mean()
        
        # Compute total cost allocation
        num_tta_triggered = cut_idx
        num_passed_through = total_classes - cut_idx
        
        estimated_time = (num_tta_triggered * retristyle_time_per_img) + \
                         (num_passed_through * baseline_time_per_img)
        
        pct_compute_saved = (1.0 - (estimated_time / (total_classes * retristyle_time_per_img))) * 100
        
        simulation_results.append({
            "classes_using_tta": num_tta_triggered,
            "pct_classes_augmented": (num_tta_triggered / total_classes) * 100,
            "simulated_accuracy": composite_accuracy,
            "estimated_runtime_sec": estimated_time,
            "compute_saved_pct": pct_compute_saved
        })
        
    return pd.DataFrame(simulation_results)
# ── Plotting ──────────────────────────────────────────────────────────────────
def _plot_gating_tradeoff(sim_df: pd.DataFrame, backbone: str, output_dir: str):
    """
    Plots the dual-axis trade-off curve between cumulative composite accuracy
    and percentage of compute time saved by gating OOD data.
    """
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()  # Create shared x-axis layout for dual indicators

    # X-axis represents the policy choice: from strict/expensive to fast/lazy
    x_metric = sim_df["pct_classes_augmented"]

    # Line 1: Composite Dataset Accuracy (Left Axis)
    line1, = ax1.plot(
        x_metric, 
        sim_df["simulated_accuracy"] * 100,  # Scale to % for readable axis
        color=COL_POS, 
        linewidth=2.5, 
        label="Simulated Accuracy"
    )
    
    # Line 2: Percentage Compute Saved (Right Axis)
    line2, = ax2.plot(
        x_metric, 
        sim_df["compute_saved_pct"], 
        color=COL_NEG, 
        linewidth=2.0, 
        linestyle="--", 
        label="Compute Saved"
    )

    # Label Formatting
    ax1.set_xlabel("% of Top-Vulnerable Classes Augmented (Gating Threshold)", fontsize=9, color="white")
    ax1.set_ylabel("Composite Accuracy (%)", fontsize=9, color=COL_POS)
    ax2.set_ylabel("Compute Time Saved (%)", fontsize=9, color=COL_NEG)

    # Style Tick Marks for Dark Mode Integration
    ax1.tick_params(axis='both', colors='white', labelsize=8)
    ax2.tick_params(axis='y', colors='white', labelsize=8)

    # Add a horizontal indicator for baseline performance (0% classes augmented)
    baseline_acc = sim_df.loc[sim_df["classes_using_tta"] == 0, "simulated_accuracy"].values[0] * 100
    ax1.axhline(baseline_acc, color=COL_MEAN, linestyle=":", alpha=0.6, label="No-TTA Baseline")

    # Combine Legends from both axes seamlessly
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center",
               framealpha=0.1, facecolor=PANEL_BG, edgecolor=SPINE_COL, fontsize=8)

    # Set boundaries cleanly
    ax1.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    
    ax1.grid(True, color="#333344", alpha=0.3, linestyle=":")

    plt.title(
        f"Adaptive Gated Inference Optimization Matrix — {backbone}\n"
        f"Accuracy Preservation vs. Computational Offloading",
        fontsize=11, fontweight="bold", color="white", pad=12
    )

    # Apply your pipeline's native visual theme styles
    TTAVisualizer.apply_dark_theme(fig)
    fig.patch.set_facecolor(DARK_BG)
    ax1.set_facecolor(PANEL_BG)
    
    # Force alignment on twin framework spines
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)

    fig.tight_layout()
    out = os.path.join(output_dir, f"{backbone}_adaptive_gating_tradeoff.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")

def _plot_scatter_grid(df, df_helped, df_hurt, gap_metrics, backbone, best_info, output_dir):
    """3 groups × 2 targets × 3 metrics = 6-panel grid per group."""
    targets = ["delta_acc", "drop"]
    groups  = [
        ("All classes",   df,        "white"),
        ("Helped by TTA", df_helped, COL_POS),
        ("Hurt by TTA",   df_hurt,   COL_NEG),
    ]

    n_rows = len(groups)
    n_cols = len(gap_metrics) * len(targets)   # 6 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    fig.suptitle(
        f"Feature-space domain gap vs TTA signals — {backbone} | "
        f"nv={best_info['nviews']}, {best_info['strategy']}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    target_labels = {"delta_acc": "Δ Accuracy (TTA)", "drop": "Accuracy drop"}

    for row_i, (group_label, df_grp, dot_color) in enumerate(groups):
        col = 0
        for metric in gap_metrics:
            for target in targets:
                ax = axes[row_i, col] if n_rows > 1 else axes[col]
                valid = df_grp[[metric, target]].dropna()

                ax.scatter(valid[metric], valid[target],
                           color=dot_color, s=16, alpha=0.55, edgecolors="none")

                if len(valid) > 2:
                    m, b = np.polyfit(valid[metric], valid[target], 1)
                    x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
                    ax.plot(x_line, m * x_line + b, color=COL_MEAN, linewidth=1.5)

                rho, p = spearmanr(valid[metric], valid[target], nan_policy="omit")
                sig = "✱" if p < 0.05 else ""
                ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
                ax.set_xlabel(FEATURE_DOMAIN_METRICS[metric], fontsize=8)
                ax.set_ylabel(target_labels[target] if col == 0 else "", fontsize=8)
                ax.set_title(
                    f"{group_label} (n={len(valid)})\n"
                    f"ρ={rho:+.3f} p={p:.3f}{sig}",
                    fontsize=8, fontweight="bold",
                )
                col += 1

    TTAVisualizer.apply_dark_theme(fig)
    fig.tight_layout()
    out = os.path.join(output_dir, f"{backbone}_scatter_grid.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def _plot_correlation_heatmap(df_strat, gap_metrics, backbone, best_info, output_dir):
    """Heatmap: rows = group × target, columns = gap metrics."""
    # Build a combined row label
    df_strat = df_strat.copy()
    df_strat["row_label"] = df_strat["group"] + " / " + df_strat["target"]

    row_order = [
        "All / delta_acc",   "All / drop",
        "Helped / delta_acc","Helped / drop",
        "Hurt / delta_acc",  "Hurt / drop",
    ]
    pivot_rho = df_strat.pivot(index="row_label", columns="gap_metric", values="rho").reindex(index=row_order, columns=gap_metrics)
    pivot_p   = df_strat.pivot(index="row_label", columns="gap_metric", values="p_value").reindex(index=row_order, columns=gap_metrics)
    pivot_n   = df_strat.pivot(index="row_label", columns="gap_metric", values="n").reindex(index=row_order, columns=gap_metrics)

    fig, ax = plt.subplots(figsize=(4 * len(gap_metrics), 1.2 * len(row_order)))
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(pivot_rho.values.astype(float), cmap=cm.get_cmap("RdYlGn"), norm=norm, aspect="auto")

    ax.set_xticks(range(len(gap_metrics)))
    ax.set_xticklabels([FEATURE_DOMAIN_METRICS[m].replace("\n", " ") for m in gap_metrics], fontsize=9, color="white")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=9, color="white")

    for i, row in enumerate(row_order):
        for j, metric in enumerate(gap_metrics):
            rho_val = pivot_rho.loc[row, metric]
            p_val   = pivot_p.loc[row, metric]
            n_val   = pivot_n.loc[row, metric]
            if pd.isna(rho_val):
                continue
            sig = "✱" if p_val < 0.05 else ""
            text_color = "white" if abs(rho_val) > 0.4 else "black"
            ax.text(j, i,
                    f"ρ={rho_val:+.2f}{sig}\np={p_val:.3f}\nn={int(n_val)}",
                    ha="center", va="center", fontsize=8,
                    color=text_color,
                    fontweight="bold" if p_val < 0.05 else "normal")

    fig.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.03, pad=0.02)
    ax.set_title(
        f"Feature-space gap vs TTA signals — {backbone}\n"
        f"nv={best_info['nviews']}, {best_info['strategy']} | ✱ p < 0.05",
        fontsize=11, fontweight="bold", color="white",
    )
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)

    out = os.path.join(output_dir, f"{backbone}_correlation_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")
def _plot_damage_zone_analysis(df: pd.DataFrame, backbone: str, output_dir: str):
    """
    Bins classes by their baseline accuracy drop and calculates the average 
    """
    # 1. Define the categorical boundaries for our damage zones
    def assign_zone(row):
        if row['drop'] < 0.30:
            return 'Low Damage\n(<30% drop)'
        elif row['drop'] < 0.60:
            return 'Mid Damage\n(30-60% drop)'
        else:
            return 'Severe Collapse\n(>60% drop)'
        
    df = df.copy()
    df['zone'] = df.apply(assign_zone, axis=1)
    
    # 2. Compute mean TTA utility per damage profile
    zone_order = ['Low Damage\n(<25% drop)', 'Mid Damage\n(25-60% drop)', 'Severe Collapse\n(>60% drop)']
    
    # Group by and extract counts and means
    stats = df.groupby('zone')['delta_acc'].agg(['mean', 'count']).reindex(zone_order)
    
    # Multiply mean by 100 to convert to readable percentage accuracy gain
    stats['mean_pct'] = stats['mean'] * 100
    
    # 3. Plotting using your theme tokens
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    # Map colors: Subdued slate, Vibrant Highlight Blue, Warning Coral Red
    colors = ['#444466', COL_POS, COL_NEG]
    
    bars = ax.bar(stats.index, stats['mean_pct'], color=colors, edgecolor=SPINE_COL, width=0.55)
    
    # Add data labels on top of the bars showing the exact gain and sample size
    for bar, (_, row) in zip(bars, stats.iterrows()):
        height = bar.get_height()
        # Handle positioning for positive/negative bars gracefully
        va_dir = 'bottom' if height >= 0 else 'top'
        xy_offset = (0, 4) if height >= 0 else (0, -12)
        
        if not pd.isna(height):
            ax.annotate(
                f"{height:+.2f}%\n(n={int(row['count'])})",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=xy_offset,
                textcoords="offset points",
                ha='center', va=va_dir,
                fontsize=8, fontweight='bold', color='white'
            )

    # Styling and formatting
    ax.set_ylabel("Average TTA Recovery Gain (Δ Accuracy %)", fontsize=9, color='white')
    ax.set_xlabel("Class Domain Degradation Profile", fontsize=9, color='white')
    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    
    ax.tick_params(axis='both', colors='white', labelsize=8)
    
    # Pad the top of the y-axis dynamically to prevent text clipping
    y_max = max(stats['mean_pct'].max() * 1.3, 2.0)
    y_min = min(stats['mean_pct'].min() * 1.3, -2.0)
    ax.set_ylim(y_min, y_max)
    
    plt.title(
        f"RetriStyle Damage Grouping — {backbone}\n",
        fontsize=11, fontweight="bold", color="white", pad=12
    )

    # Match your native dashboard infrastructure aesthetics
    TTAVisualizer.apply_dark_theme(fig)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)

    fig.tight_layout()
    out = os.path.join(output_dir, f"{backbone}_damage_zone_profile.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")

if __name__ == "__main__":
    analyse_correlation(
        baseline_path="results/ablation/tta_inference/predictions/imagenet/test_r_c26/resnet18_retristyle_vanilla_dino_nrefs1_seed71397589.json", 
        tta_dir="results/ablation/tta_inference/predictions/imagenet/test_r_c26",
        val_pred_path="results/baseline/tta_inference/predictions/imagenet/val@test_r_c26/resnet18_geometric_vanilla_nviews1_seed71397589.json",
        #baseline_path="results/baseline/test_r/tta_inference/predictions/imagenet/test_r/resnet18_geometric_vanilla_nviews1_seed265017005.json",
        #tta_dir="results/baseline/test_r/tta_inference/predictions/imagenet/test_r",        
        #val_pred_path="results/baseline/tta_inference/predictions/imagenet/val@test_r/resnet18_geometric_vanilla_nviews1_seed71397589.json",
        feature_domain_path="results/domain_gap/feature_space/test_r_c26/26/resnet18_domain_gap.json",
        backbone="resnet18",
        classifier="resnet18",
        output_dir="./figures/correlation/resnet18",
    )

    
