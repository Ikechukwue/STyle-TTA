"""
Geometric TTA Analysis for ResNet18 on ImageNet → ImageNet-R
=============================================================
Usage:
    python analyze_geo_tta.py
"""

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
from .helpers.support_funct import load_json, get_names
# ── Global Constants & Configurations ──────────────────────────────────────────

classifier = "resnet18"
method = 'geometric'
BASELINE_PATH = (
    "/home/stud/nemmler/retristyle/results/baseline/test_r/"
    "tta_inference/predictions/imagenet/test_r/"
    f"{classifier}_geometric_vanilla_nviews1_seed265017005.json"
)
TTA_DIR = (
    "/home/stud/nemmler/retristyle/results/baseline/test_r/"
    "tta_inference/predictions/imagenet/test_r"
)

DOMAIN_PATH = "/home/stud/nemmler/retristyle/results/domain_stats/imagenet_test_r/0_30_imagenet_test_r.json"
OUTPUT_DIR = f"./figures/tta_analysis_output/{method}/{classifier}"

DOMAIN_METRICS = {
    "edge_similarity": ("edge_similarity_mean", "↑ less gap"),
    "lpips": ("lpips_score_mean", "↑ more gap"),
    "ldc_haus": ("ldc_haus_mean", "↑ more gap"),
    "depth_mae": ("depthanything_v2_large_mae_mean", "↑ more gap"),
}

DOMAIN_LABEL_MAP = {
    "edge_similarity": "Edge Similarity\n(↑ less gap)",
    "lpips": "LPIPS\n(↑ more gap)",
    "ldc_haus": "LDC Hausdorff\n(↑ more gap)",
    "depth_mae": "Depth MAE\n(↑ more gap)",
}

# Theme Constants
DARK_BG = "#12121f"
PANEL_BG = "#1e1e2e"
SPINE_COL = "#444466"
COL_POS = "#4c9be8"   # Improved
COL_NEG = "#e05c5c"   # Hurt
COL_MEAN = "#f5c842"  # Metric Average


# ── 1. Infrastructure Layer (I/O) ─────────────────────────────────────────────

    
def parse_tta_filename(fname: str, method:str = "geometric") -> Optional[Dict[str, Any]]:
    """Parses config markers out of standardized file strings."""
    if method == 'geometric':
        m = re.search(
            r"([a-zA-Z0-9_]+)_geometric_([a-zA-Z0-9]+)_nviews(\d+)_seed(\d+)",
            fname
        )
    elif method == 'retristyle':
        m = re.search(
            r"([a-zA-Z0-9_]+)_retristyle_([a-zA-Z0-9]+)_dino_nrefs(\d+)_seed(\d+)",
            fname
        )
    else:
        return None
    
    if not m:
        return None
    return {
        "classifier": m.group(1),
        "eval_strategy": m.group(2),
        "nviews": int(m.group(3)),
        "seed": int(m.group(4)),
    }


def extract_class_metrics(pred_json: dict) -> pd.DataFrame:
        """Transforms JSON inference records into aggregated class-level arrays."""
        rows = []
        for p in pred_json["predictions"]:
            probs = np.array(p["y_pred"])
            is_correct = int(np.argmax(probs) == p["y_true"])
            rows.append({
                "class_id": int(p["y_true"]),
                "is_correct": is_correct,
                "confidence": float(np.max(probs)),
                "pred_entropy": float(entropy(probs)),
            })
        df = pd.DataFrame(rows)
        return (
            df.groupby("class_id")
            .agg(acc=("is_correct", "mean"),
                 conf=("confidence", "mean"),
                 ent=("pred_entropy", "mean"))
            .reset_index()
        )

def load_pipeline_dataset(baseline_path: str, tta_dir: str, classifier: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads baseline and dynamically scans directory paths for experimental arrays."""
        method = 'geometric'
        print("Loading baseline statistics...")
        base_df = extract_class_metrics(load_json(baseline_path))
        base_df = base_df.rename(columns={"acc": "base_acc", "conf": "base_conf", "ent": "base_ent"})
        print(f"  Baseline overall accuracy: {base_df['base_acc'].mean():.4f} | {len(base_df)} classes")

        method_pattern = f"{classifier}_geometric_vanilla_nviews*.json" if method == "geometric" else f"{classifier}_retristyle_vanilla_dino_nrefs*.json"
        tta_pattern = os.path.join(tta_dir, method_pattern)
        print(tta_pattern)
        tta_files = glob.glob(tta_pattern)
        print(f"\nFound {len(tta_files)} TTA prediction files matching runtime variables.")

        records = []
        for path in tta_files:
            meta = parse_tta_filename(os.path.basename(path), method)
            if meta is None:
                print(f" Could not parse metadata signature: {os.path.basename(path)}, skipping.")
                continue
            metrics = extract_class_metrics(load_json(path))
            metrics["nviews"] = meta["nviews"]
            metrics["eval_strategy"] = meta["eval_strategy"]
            metrics["seed"] = meta["seed"]
            records.append(metrics)

        if not records:
            raise RuntimeError("Halting Execution: No matching TTA validation sets could be mapped.")

        tta_raw = pd.concat(records, ignore_index=True)
        tta_avg = (
            tta_raw
            .groupby(["class_id", "nviews", "eval_strategy"])
            .agg(tta_acc=("acc", "mean"), tta_conf=("conf", "mean"), tta_ent=("ent", "mean"))
            .reset_index()
        )

        df = tta_avg.merge(base_df, on="class_id", how="inner")
        df["delta_acc"] = df["tta_acc"] - df["base_acc"]
        df["delta_conf"] = df["tta_conf"] - df["base_conf"]
        df["delta_ent"] = df["tta_ent"] - df["base_ent"]

        print(f"  Seeds tracked per config: {tta_raw.groupby(['nviews','eval_strategy'])['seed'].nunique().to_dict()}")
        return df, tta_raw

def process_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generates sorted rank groupings and finds best setting"""
    config_summary = (
        df.groupby(["nviews", "eval_strategy"])
        .agg(
            mean_delta_acc=("delta_acc", "mean"),
            mean_delta_conf=("delta_conf", "mean"),
            mean_delta_ent=("delta_ent", "mean"),
            pct_improved=("delta_acc", lambda x: (x > 0).mean() * 100),
            pct_hurt=("delta_acc", lambda x: (x < 0).mean() * 100),
        )
        .reset_index()
        .sort_values("mean_delta_acc", ascending=False)
    )

    best_row = config_summary.iloc[0]
    best_info = {
        "nviews": int(best_row["nviews"]),
        "strategy": best_row["eval_strategy"],
        "mean_delta_acc": best_row["mean_delta_acc"]
    }

    views_summary = (
        df[df["eval_strategy"] == best_info["strategy"]]
        .groupby("nviews")
        .agg(mean_delta_acc=("delta_acc", "mean"), std_delta_acc=("delta_acc", "std"))
        .reset_index()
        .sort_values("nviews")
    )

    return config_summary, views_summary, best_info

def construct_domain_features(domain_path: str) -> pd.DataFrame:
    """Parses external system structural json maps into flat tabular records."""
    print("\nLoading out-of-domain statistics map...")
    domain_raw = load_json(domain_path)["class_summaries"]
    df = pd.DataFrame([
        {
            "class_id": int(k),
            "edge_similarity": v["edge_similarity_mean"],
            "lpips": v["lpips_score_mean"],
            "ldc_haus": v["ldc_haus_mean"],
            "depth_mae": v["depthanything_v2_large_mae_mean"],
        }
        for k, v in domain_raw.items()
    ])
    print(f"  Domain features calculated across {len(df)} discrete classes.")
    return df

def run_stratified_correlation(df_dom: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits populations into cohorts and evaluates structural trends via Spearman ρ."""
    df_helped = df_dom[df_dom["delta_acc"] > 0].copy()
    df_hurt = df_dom[df_dom["delta_acc"] < 0].copy()

    def correlate_group(df_grp: pd.DataFrame, label: str) -> pd.DataFrame:
        rows = []
        for stat in DOMAIN_METRICS:
            rho, p = spearmanr(df_grp[stat], df_grp["delta_acc"], nan_policy="omit")
            rows.append({"group": label, "domain_metric": stat, "rho": rho, "p_value": p, "n": len(df_grp)})
        return pd.DataFrame(rows)

    df_corr = pd.concat([
        correlate_group(df_dom, "All"),
        correlate_group(df_helped, "Helped"),
        correlate_group(df_hurt, "Hurt")
    ], ignore_index=True)

    return df_corr, df_helped, df_hurt


# ── 4. Visual Presentation Engine (Plotting Operations) ───────────────────────

class TTAVisualizer:
    """Isolates UI configuration, visualization adjustments, and multi-axes plots."""

    @staticmethod
    def apply_dark_theme(fig: plt.Figure):
        """Uniformly maps plotting nodes over dark target aesthetics."""
        for ax in fig.axes:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors="white", labelsize=8)
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor(SPINE_COL)
        fig.patch.set_facecolor(DARK_BG)

    @classmethod
    def generate_overview_dashboard(cls, config_summary: pd.DataFrame, df_best: pd.DataFrame, 
                                    views_summary: pd.DataFrame, best_info: dict, out_path: str):
        """Generates Figure 1: Pipeline optimization metrics overview."""
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Geometric TTA Overview — {classifier.upper()} | ImageNet → ImageNet-R", 
                     fontsize=13, fontweight="bold", y=0.99)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

        # Plot A: Config ranking
        ax_a = fig.add_subplot(gs[0, 0])
        labels = [f"nv={r.nviews}\n{r.eval_strategy}" for _, r in config_summary.iterrows()]
        colors = [COL_NEG if v < 0 else COL_POS for v in config_summary["mean_delta_acc"]]
        bars = ax_a.barh(labels, config_summary["mean_delta_acc"], color=colors, edgecolor="white", height=0.6)
        ax_a.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_a.set_xlabel("Mean Δ Accuracy", fontsize=9)
        ax_a.set_title("A Config Ranking", fontweight="bold", fontsize=10)
        ax_a.invert_yaxis()
        for bar, val in zip(bars, config_summary["mean_delta_acc"]):
            ax_a.text(val + (0.0002 if val >= 0 else -0.0002), bar.get_y() + bar.get_height() / 2,
                      f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=7, color="white")

        # Plot B: Distribution
        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.hist(df_best["delta_acc"], bins=40, color=COL_POS, edgecolor="white", linewidth=0.4)
        ax_b.axvline(0, color=COL_NEG, linewidth=1.2, linestyle="--", label="no change")
        ax_b.axvline(df_best["delta_acc"].mean(), color=COL_MEAN, linewidth=1.2, label=f"mean={df_best['delta_acc'].mean():+.4f}")
        ax_b.set_xlabel("Δ Accuracy per Class", fontsize=9)
        ax_b.set_ylabel("# Classes", fontsize=9)
        ax_b.set_title(f"B Δ Acc Distribution\n(nv={best_info['nviews']}, {best_info['strategy']})", fontweight="bold", fontsize=10)
        ax_b.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="white")

        # Plot C: Top/Bottom
        ax_c = fig.add_subplot(gs[0, 2])
        nmb = 20 if len(df_best) >=40 else 5 
        combined = pd.concat([df_best.head(nmb), df_best.tail(nmb)]).drop_duplicates("class_id").sort_values("delta_acc")

        name_list = get_names()
        y_labels = [name_list[cid] for cid in combined["class_id"]]

        ax_c.barh(y_labels, combined["delta_acc"], 
                  color=[COL_NEG if v < 0 else COL_POS for v in combined["delta_acc"]], edgecolor="white", height=0.7)
        ax_c.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax_c.set_xlabel("Δ Accuracy", fontsize=9)
        ax_c.set_title("C Top/Bottom 20 Classes", fontweight="bold", fontsize=10)
        ax_c.tick_params(axis="y", labelsize=6)

        # Plot D: Views Scaling
        ax_d = fig.add_subplot(gs[1, 0])
        ax_d.plot(views_summary["nviews"], views_summary["mean_delta_acc"], marker="o", color=COL_POS, linewidth=2, markersize=6)
        ax_d.fill_between(views_summary["nviews"], views_summary["mean_delta_acc"] - views_summary["std_delta_acc"],
                          views_summary["mean_delta_acc"] + views_summary["std_delta_acc"], alpha=0.15, color=COL_POS)
        ax_d.axhline(0, color=COL_NEG, linewidth=0.8, linestyle="--")
        ax_d.set_xlabel("N Views", fontsize=9)
        ax_d.set_ylabel("Mean Δ Accuracy", fontsize=9)
        ax_d.set_title(f"D Views Scaling ({best_info['strategy']})", fontweight="bold", fontsize=10)

        # Plot E: Scatter Base Accuracy
        ax_e = fig.add_subplot(gs[1, 1])
        sc = ax_e.scatter(df_best["base_acc"], df_best["delta_acc"], c=df_best["delta_acc"], cmap="RdYlGn", s=14, alpha=0.65, vmin=-0.5, vmax=0.5)
        ax_e.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        rho, p = spearmanr(df_best["base_acc"], df_best["delta_acc"], nan_policy="omit")
        ax_e.set_xlabel("Baseline Accuracy", fontsize=9)
        ax_e.set_ylabel("Δ Accuracy", fontsize=9)
        ax_e.set_title(f"E Baseline vs Δ Accuracy\nρ={rho:+.3f} p={p:.4f}", fontweight="bold", fontsize=10)
        fig.colorbar(sc, ax=ax_e, label="Δ Acc", fraction=0.04)

        # Plot F: Entropy
        ax_f = fig.add_subplot(gs[1, 2])
        sc2 = ax_f.scatter(df_best["delta_ent"], df_best["delta_acc"], c=df_best["base_acc"], cmap="viridis", s=14, alpha=0.65)
        ax_f.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
        ax_f.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
        rho_ef, p_ef = spearmanr(df_best["delta_ent"], df_best["delta_acc"], nan_policy="omit")
        ax_f.set_xlabel("Δ Entropy (↓ = less confused)", fontsize=9)
        ax_f.set_ylabel("Δ Accuracy", fontsize=9)
        ax_f.set_title(f"F Δ Entropy vs Δ Accuracy\nρ={rho_ef:+.3f} p={p_ef:.4f}", fontweight="bold", fontsize=10)
        fig.colorbar(sc2, ax=ax_f, label="Base Acc", fraction=0.04)

        cls.apply_dark_theme(fig)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @classmethod
    def generate_stratified_plots(cls, groups: list, domain_stats: list, out_path: str, best_info: dict):
        """Generates Figure 2: Exhaustive 3x4 grid displaying correlation scatter trends."""
        n_rows, n_cols = len(groups), len(domain_stats)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle(f"Domain Gap vs Δ Accuracy — {classifier.upper()} | Best Config: nv={best_info['nviews']}, {best_info['strategy']}",
                     fontsize=13, fontweight="bold", y=1.01)

        for row_i, (group_label, df_grp, dot_color, n_label) in enumerate(groups):
            for col_j, stat in enumerate(domain_stats):
                ax = axes[row_i, col_j] if n_rows > 1 else axes[col_j]
                ax.scatter(df_grp[stat], df_grp["delta_acc"], color=dot_color, s=18, alpha=0.55, edgecolors="none")

                valid = df_grp[[stat, "delta_acc"]].dropna()
                if len(valid) > 2:
                    m, b = np.polyfit(valid[stat], valid["delta_acc"], 1)
                    x_line = np.linspace(valid[stat].min(), valid[stat].max(), 100)
                    ax.plot(x_line, m * x_line + b, color=COL_MEAN, linewidth=1.5, alpha=0.9)

                rho, p = spearmanr(df_grp[stat], df_grp["delta_acc"], nan_policy="omit")
                sig_marker = "✱" if p < 0.05 else ""
                ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
                ax.set_xlabel(DOMAIN_LABEL_MAP[stat], fontsize=8)
                ax.set_ylabel("Δ Accuracy" if col_j == 0 else "", fontsize=8)
                ax.set_title(f"{group_label} {n_label}\nρ={rho:+.3f} p={p:.3f} {sig_marker}", fontsize=8.5, fontweight="bold")

        cls.apply_dark_theme(fig)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def generate_heatmap(df_corr: pd.DataFrame, domain_stats: list, out_path: str, best_info: dict):
        """Generates Figure 3: Summary matrix correlation metrics heatmap."""
        heatmap_data = df_corr.pivot(index="group", columns="domain_metric", values="rho")
        heatmap_p = df_corr.pivot(index="group", columns="domain_metric", values="p_value")

        row_order = ["All", "Helped", "Hurt"]
        heatmap_data = heatmap_data.reindex(index=row_order, columns=domain_stats)
        heatmap_p = heatmap_p.reindex(index=row_order, columns=domain_stats)

        fig, ax = plt.subplots(figsize=(8, 4))
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(heatmap_data.values.astype(float), cmap=cm.get_cmap('RdYlGn'), norm=norm, aspect="auto")

        ax.set_xticks(range(len(domain_stats)))
        ax.set_xticklabels([DOMAIN_LABEL_MAP[s].replace("\n", " ") for s in domain_stats], fontsize=9, color="white")
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels(row_order, fontsize=10, color="black")

        for i, row in enumerate(row_order):
            for j, stat in enumerate(domain_stats):
                rho_val = heatmap_data.loc[row, stat]
                p_val = heatmap_p.loc[row, stat]
                n_val = df_corr[(df_corr["group"] == row) & (df_corr["domain_metric"] == stat)]["n"].values[0]
                sig = "✱" if p_val < 0.05 else ""
                
                cell_text = f"ρ={rho_val:+.2f}{sig}\np={p_val:.3f}\nn={n_val}"
                text_color = "white" if abs(rho_val) > 0.4 else "black"
                ax.text(j, i, cell_text, ha="center", va="center", fontsize=8,
                        color=text_color, fontweight="bold" if p_val < 0.05 else "normal")

        fig.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.03, pad=0.02)
        ax.set_title(f"Domain Gap vs Δ Accuracy — Spearman ρ Summary\n{classifier.upper()} | nv={best_info['nviews']}, {best_info['strategy']} | ✱ p < 0.05",
                     fontsize=11, fontweight="bold", color="white")
        
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COL)
        ax.tick_params(colors="white")
        
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ── 5. Operational Workflow Orchestration ──────────────────────────────────────



def create_summary(classifier, baseline_path, tta_dir, domain_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Ingestion & Metric Calculations
    df, _ = load_pipeline_dataset(baseline_path, tta_dir, classifier)
    config_summary, views_summary, best_info = process_summaries(df)

    print("\n── Config Ranking (by mean Δ accuracy) ──────────────────────────────")
    print(config_summary.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nBest config: nviews={best_info['nviews']}, eval_strategy={best_info['strategy']} (mean Δacc={best_info['mean_delta_acc']:+.4f})")

    # Filter baseline mappings around elite parameters
    df_best = df[(df["nviews"] == best_info["nviews"]) & (df["eval_strategy"] == best_info["strategy"])].copy()
    df_best = df_best.sort_values("delta_acc", ascending=False).reset_index(drop=True)

    # Domain Feature Ingestion & Statistical Stratification
    domain_df = construct_domain_features(DOMAIN_PATH)
    df_dom = df_best.merge(domain_df, on="class_id", how="inner")
    
    print(f"  Classes after merge: {len(df_dom)} (helped={(df_dom['delta_acc'] > 0).sum()}, hurt={(df_dom['delta_acc'] < 0).sum()})")
    df_corr, df_helped, df_hurt = run_stratified_correlation(df_dom)

    print("\n── Domain × Δ Accuracy Correlations ─────────────────────────────────")
    print(df_corr.to_string(index=False, float_format="{:.4f}".format))

    # Persistence Save Operations
    config_summary.to_csv(os.path.join(output_dir, "config_ranking.csv"), index=False)
    df_best.to_csv(os.path.join(output_dir, "class_deltas_best_config.csv"), index=False)
    views_summary.to_csv(os.path.join(output_dir, "views_scaling_summary.csv"), index=False)
    df_corr.to_csv(os.path.join(output_dir, "domain_correlations_stratified.csv"), index=False)
    df_dom.to_csv(os.path.join(output_dir, "class_deltas_with_domain.csv"), index=False)

    # Graphical Generation Engine Execution Tasks
    print("\nGenerating analytical summary dashboards...")
    
    TTAVisualizer.generate_overview_dashboard(
        config_summary, df_best, views_summary, best_info, 
        os.path.join(output_dir, "fig1_tta_overview.png")
    )
    
    cohort_groups = [
        ("All classes", df_dom, "white", f"n={len(df_dom)}"),
        ("Helped by TTA", df_helped, COL_POS, f"n={len(df_helped)}"),
        ("Hurt by TTA", df_hurt, COL_NEG, f"n={len(df_hurt)}"),
    ]
    domain_cols = list(DOMAIN_LABEL_MAP.keys())
    
    TTAVisualizer.generate_stratified_plots(
        cohort_groups, domain_cols, 
        os.path.join(output_dir, "fig2_domain_correlations.png"), best_info
    )
    
    TTAVisualizer.generate_heatmap(
        df_corr, domain_cols, 
        os.path.join(output_dir, "fig3_correlation_heatmap.png"), best_info
    )

    print(f"\nComplete evaluation successful! Structural updates exported to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    create_summary(classifier=classifier, 
                   baseline_path=BASELINE_PATH,
                   tta_dir=TTA_DIR,
                   domain_path=DOMAIN_PATH,
                   output_dir=OUTPUT_DIR, 
                    )
