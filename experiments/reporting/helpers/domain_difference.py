from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
from .support_funct import get_names
import json
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.colors as mcolors
import matplotlib.cm as cm

METRIC_DIRECTIONS = {
    "ssim_mean": " (↑)",
    "lpips_score_mean": " (↓)",  # LPIPS is a distance/error score; lower is better!
    "histogramm_distance_mean": " (↓)",
    "color_moment_distance_mean": " (↓)",
    "edge_similarity_mean": " (↑)",
    "ldc_dists_mean": " (↓)",
    "ldc_fom_mean": " (↑)",      # Figure of merit; higher is better
    "depthanything_v2_large_mae_mean": " (↓)", # Mean Absolute Error; lower is better
    "dpt_large_mae_mean": " (↓)",
    "depthanything_v2_large_spear_mean": " (↑)", # Spearman correlation; higher is better
    "dpt_large_spear_mean": " (↑)"
}

def plot_domain_shift_analysis(data_baseline: dict, data_shifted: dict, output_dir: Path, args):
    """
    Generates a publication-quality grid separating domain shift 
    metrics into categorical groups, showing baseline vs shifted side-by-side.
    """
    stats_base = data_baseline.get("global_summary", {})
    stats_shift = data_shifted.get("global_summary", {})
    
    if not stats_base or not stats_shift:
        print("  [skip] Missing global summaries for comparison.")
        return

    metric_groups = {
        "Perceptual & Structure": [
            ("SSIM", "ssim_mean", "ssim_std"),
            ("LPIPS Score", "lpips_score_mean", "lpips_score_std"),
        ],
        "Color & Distribution": [
            ("Histogram Dist.", "histogramm_distance_mean", "histogramm_distance_std"),
            ("Color Moment Dist.", "color_moment_distance_mean", "color_moment_distance_std"),
        ],
        "Edges & Boundaries": [
            ("Edge Similarity", "edge_similarity_mean", "edge_similarity_std"),
            ("LDC Distance", "ldc_dists_mean", "ldc_dists_std"),
            ("LDC Figure of Merit", "ldc_fom_mean", "ldc_fom_std"),
        ],
        "Depth & Geometry": [
            ("DepthAnything MAE", "depthanything_v2_large_mae_mean", "depthanything_v2_large_mae_std"),
            ("DPT MAE", "dpt_large_mae_mean", "dpt_large_mae_std"),
            ("DepthAnything Spear.", "depthanything_v2_large_spear_mean", "depthanything_v2_large_spear_std"),
            ("DPT Spearman", "dpt_large_spear_mean", "dpt_large_spear_std"),
        ]
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()
    colors = plt.get_cmap("Set2").colors

    for idx, (group_name, metrics) in enumerate(metric_groups.items()):
        ax = axes[idx]
        
        labels = []
        means_base, stds_base = [], []
        means_shift, stds_shift = [], []
        
        for name, mean_k, std_k in metrics:
            if mean_k in stats_base and mean_k in stats_shift:
                direction_arrow = METRIC_DIRECTIONS.get(mean_k, "")
                labels.append(f"{name}{direction_arrow}")

                means_base.append(stats_base[mean_k])
                stds_base.append(stats_base.get(std_k, 0))
                means_shift.append(stats_shift[mean_k])
                stds_shift.append(stats_shift.get(std_k, 0))

        if not labels:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue

        y_pos = np.arange(len(labels))
        bar_height = 0.35  # Dynamic width scaling to prevent overlapping
        
        # Plot Baseline Bars
        bars_base = ax.barh(y_pos - bar_height/2, means_base, xerr=stds_base, 
                            height=bar_height, align='center', alpha=0.5, 
                            color='gray', edgecolor='none', capsize=4, label='Baseline')
        
        # Plot Shifted Domain Bars
        bars_shift = ax.barh(y_pos + bar_height/2, means_shift, xerr=stds_shift, 
                             height=bar_height, align='center', alpha=0.9, 
                             color=colors[:len(labels)], edgecolor='none', capsize=4, label='Shifted')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()  # Top-down tracking order
        ax.set_title(group_name, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlim(0, 1.15)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)

        # Print inline value metrics cleanly on bars
        for b_base, b_shift in zip(bars_base, bars_shift):
            w_base = b_base.get_width()
            w_shift = b_shift.get_width()
            ax.text(w_base + 0.01, b_base.get_y() + b_base.get_height()/2, f'{w_base:.2f}', 
                    ha='left', va='center', fontsize=8, color='dimgray')
            ax.text(w_shift + 0.01, b_shift.get_y() + b_shift.get_height()/2, f'{w_shift:.2f}', 
                    ha='left', va='center', fontsize=8, fontweight='bold')

    fig.suptitle("Domain Shift Metric Analysis Baseline Comparison Summary", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.15, right=0.95, hspace=0.25, wspace=0.25)
    
    out = output_dir / "domain_shift_metrics_summary.pdf"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [saved] {out}")

def plot_domain_shift_class_scatter_per_group(data_baseline: dict, data_shifted: dict, output_dir: Path, args=None):
    """
    Generates class-wise scatter plots for domain shift metrics splitting each metric
    into Baseline vs Shifted visual pairs sharing the same coordinate mapping.
    """
    summaries_base = data_baseline.get("class_summaries", {})
    summaries_shift = data_shifted.get("class_summaries", {})
    name_list = get_names(args.split)
    
    if not summaries_base or not summaries_shift:
        print("  [skip] Comprehensive multi-set class summary datasets missing")
        return

    metric_groups = {
        "perceptual_structure": ("Perceptual & Structure Metrics", [("ssim_mean", "SSIM"), ("lpips_score_mean", "LPIPS Score")]),
        "color_distribution": ("Color & Distribution Metrics", [("histogramm_distance_mean", "Histogram Dist."), ("color_moment_distance_mean", "Color Moment Dist.")]),
        "edges_boundaries": ("Edges & Boundaries Metrics", [("edge_similarity_mean", "Edge Similarity"), ("ldc_dists_mean", "LDC Distance"), ("ldc_fom_mean", "LDC Figure of Merit")]),
        "depth_geometry": ("Depth & Geometry Metrics", [("depthanything_v2_large_mae_mean", "DepthAnything MAE"), ("dpt_large_mae_mean", "DPT MAE"), ("depthanything_v2_large_spear_mean", "DepthAnything Spear."), ("dpt_large_spear_mean", "DPT Spearman")])
    }

    # Restructure data dictionaries for sequential reading
    def parse_set(source):
        parsed = {}
        for class_id, metrics in source.items():
            for k, val in metrics.items():
                if k.endswith("_mean"):
                    parsed.setdefault(k, []).append((class_id, val))
        return parsed

    classes_base = parse_set(summaries_base)
    classes_shift = parse_set(summaries_shift)

    unique_classes = sorted(list(summaries_base.keys()), key=int)
    color_map = mpl.colormaps["tab20"].resampled(max(len(unique_classes), 2))
    class_colors = {cls: color_map(i) for i, cls in enumerate(unique_classes)}

    subfolder_dir = output_dir / "domain_shift_groups"
    subfolder_dir.mkdir(parents=True, exist_ok=True)
    multipage_pdf_path = output_dir / "domain_shift_class_scatter_multipage.pdf"
    
    print(f"  [processing] Splitting charts into baseline vs shift paired structures...")

    with PdfPages(multipage_pdf_path) as pdf:
        for group_id, (group_name, metrics_list) in metric_groups.items():
            # Height expanded dynamically based on metric list scale to avoid tight clipping warnings
            fig, ax = plt.subplots(figsize=(11, len(metrics_list) * 2.2))
            
            y_ticks = []
            y_labels = []
            y_counter = 0
            
            for metric_key, display_name in metrics_list:
                if metric_key not in classes_base or metric_key not in classes_shift:
                    continue
                
                direction_arrow = METRIC_DIRECTIONS.get(metric_key, "")
                # --- LANE 1: Baseline Generation Row ---
                y_ticks.append(y_counter)
                y_labels.append(f"{display_name}{direction_arrow}\n(Base)")
                ax.axhline(y_counter, color='gray', linestyle=':', alpha=0.2, zorder=1)
                
                for class_id, val in classes_base[metric_key]:
                    np.random.seed(int(class_id) + 42)
                    jitter = np.random.uniform(-0.08, 0.08)
                    ax.scatter(val, y_counter + jitter, color=class_colors[class_id], 
                               edgecolor='black', linewidth=0.5, s=70, alpha=0.4, marker='o', zorder=2)

                # --- LANE 2: Shifted Generation Row ---
                y_ticks.append(y_counter + 1)
                y_labels.append(f"{display_name}\n(Shift)")
                ax.axhline(y_counter + 1, color='gray', linestyle=':', alpha=0.2, zorder=1)
                
                for class_id, val in classes_shift[metric_key]:
                    np.random.seed(int(class_id) + 42)
                    jitter = np.random.uniform(-0.08, 0.08)
                    class_name = name_list[int(class_id)]
                    ax.scatter(val, y_counter + 1 + jitter, color=class_colors[class_id], 
                               edgecolor='black', linewidth=0.7, s=80, alpha=0.9, marker='s', zorder=2,
                               label=class_name if y_counter == 0 else "")

                # Insert dark separating barrier between metric blocks
                ax.axhline(y_counter + 1.6, color='black', linestyle='-', alpha=0.15)
                y_counter += 2.5

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=9.5)
            ax.set_ylim(-0.7, y_counter - 1.2)
            ax.invert_yaxis()
            ax.set_xlim(-0.05, 1.05)
            ax.set_xlabel("Globally Normalized Metric Space [0, 1]", fontsize=10, labelpad=8)
            ax.set_title(group_name, fontsize=13, fontweight="bold", pad=15)
            ax.grid(axis='x', linestyle='--', alpha=0.5)

            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', 
                          frameon=True, fontsize=8.5, bbox_to_anchor=(1.18, 1.0))

            fig.subplots_adjust(top=0.88, bottom=0.15, left=0.24, right=0.85)
            pdf.savefig(fig, bbox_inches='tight')
            
            individual_out = subfolder_dir / f"domain_shift_{group_id}.pdf"
            fig.savefig(individual_out, bbox_inches='tight')
            plt.close(fig)

    print(f"  [saved multi-page] {multipage_pdf_path}")


def plot_domain_shift_class_rankings_by_group(data:dict, output_dir: Path, args=None):
    """
    Generates isolated, standalone PDF files for each metric group saved inside a 
    dedicated subfolder pipeline. Parallel columns list class names ordered from best to worst.
    
    The top 20 classes from the FIRST metric column are tracked across all subsequent 
    metric columns on that page using matching color fills for easy visual scanning.
    """

    class_summaries = data.get("class_summaries", {})
    if not class_summaries:
        print("  [skip] No class-wise summaries available for rankings.")
        return

    split_str = args.split if args else "test_r"
    try:
        name_list = get_names(split_str)
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
