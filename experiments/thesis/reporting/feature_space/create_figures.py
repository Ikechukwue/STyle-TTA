import matplotlib.pyplot as plt
from pathlib import Path
from config.constants import ALL_CLASSIFIERS
from experiments.thesis.reporting.printing.ratios import get_domain_gap_class, get_class_acc
import numpy as np 

def plot_accuracy_drop_vs_metric(metrics=["mmd", "wasserstein", "kl_symmetric", "gram_distance"], output_path="./figures/domain_comparison"):
    print("Loading data...")
    class_metrics = get_domain_gap_class()
    class_acc = get_class_acc()

    if not class_metrics or not class_acc:
        print("Error: Required data dictionaries are missing or empty.")
        return

    for target_metric in metrics:
        out_file = Path(output_path) / f"accuracy_drop_vs_{target_metric}.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        plotted_count = 0

        for i, cls in enumerate(ALL_CLASSIFIERS):
            if i >= len(axes):
                break
            
            ax = axes[i]
            if cls not in class_acc:
                ax.set_title(f"{cls} (Missing Acc)")
                continue

            class_pairs = []
            for c in class_acc[cls]:
                if c not in class_metrics:
                    continue

                delta_acc = class_acc[cls][c]["delta_acc"]
                if target_metric == "gram_distance":
                    metric_val = class_metrics[c].get("gram_distance", 0.0)
                else:
                    metric_val = class_metrics[c].get(cls, {}).get(target_metric, 0.0)
                    
                class_pairs.append((delta_acc, metric_val))

            if not class_pairs:
                ax.set_title(f"{cls} (No Data)")
                continue

            class_pairs.sort(key=lambda x: x[0])
            sorted_deltas = np.array([p[0] for p in class_pairs])
            sorted_metrics = np.array([p[1] for p in class_pairs])

            # Raw Data Scatter
            ax.scatter(sorted_deltas, sorted_metrics, alpha=0.5, s=15, color=f"C{i}", label="Classes")
            
            # Linear Regression Calculation
            slope, intercept = np.polyfit(sorted_deltas, sorted_metrics, 1)
            y_pred = slope * sorted_deltas + intercept
            
            # R² Calculation to evaluate goodness of fit
            y_bar = np.mean(sorted_metrics)
            ss_tot = np.sum((sorted_metrics - y_bar) ** 2)
            ss_res = np.sum((sorted_metrics - y_pred) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Plot Regression Line
            ax.plot(sorted_deltas, y_pred, color="black", linestyle="-", linewidth=1.5, label="Fit", a)
            
            # Panel Annotations
            ax.set_title(f"{cls}", fontsize=12, fontweight="bold")
            ax.text(
                0.05, 0.92, 
                f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r2:.3f}", 
                transform=ax.transAxes, 
                fontsize=10, 
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
            )
            ax.grid(True, linestyle="--", alpha=0.5)
            
            if i == 0:  # Only add legend to the first subplot to save space
                ax.legend(loc="lower right")
                
            plotted_count += 1
            print(f"Subplot populated with trendline for: {cls}")

        if plotted_count == 0:
            print("Error: No data was plotted. Saving aborted.")
            plt.close()
            return

        fig.text(0.5, 0.04, "Class Accuracy Drop (Base Acc - OOD Acc)", ha="center", fontsize=14, fontweight="bold")
        fig.text(0.04, 0.5, f"Class {target_metric.replace('_', ' ').title()}", va="center", rotation="vertical", fontsize=14, fontweight="bold")
        fig.suptitle(f"Linear Trend Profiles: {target_metric.replace('_', ' ').title()} vs Performance Drop", fontsize=16, fontweight="bold", y=0.96)
        
        plt.tight_layout(rect=[0.05, 0.06, 0.95, 0.93])
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Success! Faceted panel grid with trendlines saved to {out_file.resolve()}")

if __name__ == "__main__":
    plot_accuracy_drop_vs_metric()
