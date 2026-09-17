import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from code.experiments.thesis.reporting.feature_space.report_feature_diff_class import version_domain_gap
def calculate_correlations(df: pd.DataFrame, method: str = "spearman", class_col: str = "class_id") -> pd.DataFrame:
    """Calculates average inter-class correlations between domain gap metrics and prediction metrics."""
    pred_cols = ["acc", "conf", "ent"]
    gap_cols = ["base_val", "delta", "k_val"]
    
    correlations = []
    
    # Group by class to calculate correlations within each class
    grouped = df.groupby(["classifier", "metric", class_col])
    
    for (cl, metric, class_val), group in grouped:
        for gap_col in gap_cols:
            for pred_col in pred_cols:
                valid_data = group[[gap_col, pred_col]].dropna()
                
                corr_val = valid_data[gap_col].corr(valid_data[pred_col], method=method) if len(valid_data) > 1 else None
                
                correlations.append({
                    "classifier": cl,
                    "gap_metric": metric,
                    "class": class_val,
                    "gap_type": gap_col,
                    "pred_metric": pred_col,
                    f"{method}_corr": corr_val
                })
                
    class_corr_df = pd.DataFrame(correlations)
    
    # Aggregate across classes by averaging the within-class correlations
    group_cols = ["classifier", "gap_metric", "gap_type", "pred_metric"]
    avg_corr_df = (
        class_corr_df
        .groupby(group_cols, as_index=False)[f"{method}_corr"]
        .mean()
    )
    
    return avg_corr_df

def plot_classifier_correlation_heatmaps(
    df: pd.DataFrame,
    gap_type: str = "delta",
    method: str = "spearman",
    save_path: str = "figures/domain_comparison/domain_corr_heatmap.png",
):
    """Generates multi-panel correlation heatmaps (Classifier vs Gap Metric)

    pivoted by prediction outcome metrics (acc, conf, ent).
    """
    corr_df = calculate_correlations(df, method=method)

    # Filter for target domain gap type (e.g., 'delta')
    subset = corr_df[corr_df["gap_type"] == gap_type].copy()

    pred_metrics = ["acc", "conf", "ent"]
    metric_titles = {
        "acc": "Accuracy Drop (acc)",
        "conf": "Confidence Drop (conf)",
        "ent": "Uncertainty Rise (ent)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, pred_metric in enumerate(pred_metrics):
        metric_data = subset[subset["pred_metric"] == pred_metric]

        pivot_matrix = metric_data.pivot(
            index="classifier",
            columns="gap_metric",
            values=f"{method}_corr",
        )

        sns.heatmap(
            pivot_matrix,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            vmin=-1.0,
            vmax=1.0,
            ax=axes[idx],
            cbar=(idx == 2),
            cbar_kws={"label": f"{method.capitalize()} Correlation (\\rho)"},
        )

        axes[idx].set_title(
            f"Correlation vs {metric_titles.get(pred_metric, pred_metric)}"
        )
        axes[idx].set_xlabel("Feature Gap Metric")
        if idx == 0:
            axes[idx].set_ylabel("Classifier")
        else:
            axes[idx].set_ylabel("")

    plt.suptitle(
        f"Classifier Sensitivity to Domain Gap Metrics (Gap Type: {gap_type})",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

if __name__ == "__main__":
    df = version_domain_gap(force=False)


    plot_classifier_correlation_heatmaps(df, gap_type="delta", method="spearman")

