"""
Geometric TTA Analysis for ResNet18 on ImageNet → ImageNet-R
=============================================================

Pipeline:
  1. Load baseline + all geometric TTA prediction files for resnet18
  2. Average across seeds → per-class metrics per (nviews, eval_strategy) config
  3. Find best config by mean Δ accuracy
  4. Class-level analysis: who benefits, sorted by Δ accuracy
  5. Views scaling: how Δ accuracy changes with nviews per class
  6. [HOOK] Domain stat correlations — plug in domain JSON here

Usage:
    python analyze_geo_tta.py
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import entropy, spearmanr
from typing import Optional, Dict, Any, Tuple

# ── Config ─────────────────────────────────────────────────────────────────────

CLASSIFIER = "resnet18"
BASELINE_PATH = (
    "/home/stud/nemmler/retristyle/results/baseline/test_r/"
    "tta_inference/predictions/imagenet/test_r/"
    f"{CLASSIFIER}_geometric_vanilla_nviews1_seed265017005.json"
)
TTA_DIR = (
    "/home/stud/nemmler/retristyle/results/baseline/test_r/"
    "tta_inference/predictions/imagenet/test_r/"
)
OUTPUT_DIR = "tta_analysis_output"


# ── I/O and Processing Helpers ─────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_class_metrics(pred_json: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per class containing aggregated metrics:
    acc, conf, ent
    """
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


def parse_tta_filename(fname: str) -> Optional[Dict[str, Any]]:
    """
    Expects: {classifier}_geometric_{eval_strategy}_nviews{N}_seed{S}.json
    Returns dict or None.
    """
    m = re.search(
        r"([a-zA-Z0-9_]+)_geometric_([a-zA-Z0-9]+)_nviews(\d+)_seed(\d+)",
        fname
    )
    if not m:
        return None
    return {
        "classifier": m.group(1),
        "eval_strategy": m.group(2),
        "nviews": int(m.group(3)),
        "seed": int(m.group(4)),
    }


# ── Pipeline Stages ────────────────────────────────────────────────────────────

def load_and_merge_data(baseline_path: str, tta_dir: str, classifier: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads baseline and all matching TTA files, processing them into merged DataFrames."""
    print("Loading baseline...")
    baseline_df = extract_class_metrics(load_json(baseline_path))
    baseline_df = baseline_df.rename(columns={"acc": "base_acc", "conf": "base_conf", "ent": "base_ent"})
    print(f"  Baseline overall accuracy: {baseline_df['base_acc'].mean():.4f}  |  {len(baseline_df)} classes")

    tta_pattern = os.path.join(tta_dir, f"{classifier}_geometric_tpt_nviews*.json")
    tta_files = glob.glob(tta_pattern)
    print(f"\nFound {len(tta_files)} TTA prediction files for {classifier}.")

    records = []
    for path in tta_files:
        meta = parse_tta_filename(os.path.basename(path))
        if meta is None:
            print(f"  ⚠️  Could not parse filename: {os.path.basename(path)}, skipping.")
            continue

        metrics = extract_class_metrics(load_json(path))
        metrics["nviews"] = meta["nviews"]
        metrics["eval_strategy"] = meta["eval_strategy"]
        metrics["seed"] = meta["seed"]
        records.append(metrics)

    if not records:
        raise RuntimeError("No TTA files loaded — check TTA_DIR and filename pattern.")

    tta_raw = pd.concat(records, ignore_index=True)
    
    # Average across seeds -> per-class per-config
    tta_avg = (
        tta_raw
        .groupby(["class_id", "nviews", "eval_strategy"])
        .agg(tta_acc=("acc", "mean"),
             tta_conf=("conf", "mean"),
             tta_ent=("ent", "mean"))
        .reset_index()
    )

    # Merge with baseline and compute metrics
    df = tta_avg.merge(baseline_df, on="class_id", how="inner")
    df["delta_acc"] = df["tta_acc"] - df["base_acc"]
    df["delta_conf"] = df["tta_conf"] - df["base_conf"]
    df["delta_ent"] = df["tta_ent"] - df["base_ent"]

    print(f"  Seeds found per config: {tta_raw.groupby(['nviews','eval_strategy'])['seed'].nunique().to_dict()}")
    return df, tta_raw


def generate_summaries(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Generates configuration and view-scaling summary reports."""
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

    best_cfg = config_summary.iloc[0]
    best_info = {
        "nviews": int(best_cfg["nviews"]),
        "strategy": best_cfg["eval_strategy"],
        "mean_delta_acc": best_cfg["mean_delta_acc"]
    }

    # Filter to view scaling metrics using the best strategy found
    df_views = df[df["eval_strategy"] == best_info["strategy"]].copy()
    views_summary = (
        df_views.groupby("nviews")
        .agg(mean_delta_acc=("delta_acc", "mean"),
             std_delta_acc=("delta_acc", "std"))
        .reset_index()
        .sort_values("nviews")
    )

    return config_summary, best_info, views_summary


def run_domain_hook_analysis(df_best: pd.DataFrame, json_path: str = "path/to/domain_stats.json"):
    """Optional Hook function for running external domain correlations."""
    if not os.path.exists(json_path):
        return
    
    print(f"\n── Running Domain Stats Hook Analysis using {json_path} ──")
    with open(json_path) as f:
        domain_json = json.load(f)["class_summaries"]
        
    domain_df = pd.DataFrame([
        {"class_id": int(k),
         "edge_similarity": v["edge_similarity_mean"],
         "ldc_haus": v["ldc_haus_mean"],
         "lpips": v["lpips_score_mean"],
         "depth_mae": v["depthanything_v2_large_mae_mean"]}
        for k, v in domain_json.items()
    ])
    df_best_dom = df_best.merge(domain_df, on="class_id", how="inner")

    for stat in ["edge_similarity", "ldc_haus", "lpips", "depth_mae"]:
        rho, p = spearmanr(df_best_dom[stat], df_best_dom["delta_acc"], nan_policy="omit")
        print(f"{stat:20s}  ρ={rho:+.3f}  p={p:.4f}")


# ── Plotting Engine ────────────────────────────────────────────────────────────

def plot_tta_analysis(
    classifier: str, 
    config_summary: pd.DataFrame, 
    df_best: pd.DataFrame, 
    views_summary: pd.DataFrame, 
    best_info: dict, 
    output_path: str
):
    """Generates and saves the analytical summary dashboard plot."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Geometric TTA Analysis — {classifier.upper()}  |  ImageNet → ImageNet-R",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Plot A: Config ranking bar chart
    ax_a = fig.add_subplot(gs[0, 0])
    labels = [f"nv={r.nviews}\n{r.eval_strategy}" for _, r in config_summary.iterrows()]
    colors = ["#e05c5c" if v < 0 else "#4c9be8" for v in config_summary["mean_delta_acc"]]
    bars = ax_a.barh(labels, config_summary["mean_delta_acc"], color=colors, edgecolor="white", height=0.6)
    ax_a.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.set_xlabel("Mean Δ Accuracy", fontsize=9)
    ax_a.set_title("A  Config Ranking", fontweight="bold", fontsize=10)
    ax_a.invert_yaxis()
    for bar, val in zip(bars, config_summary["mean_delta_acc"]):
        ax_a.text(
            val + (0.0002 if val >= 0 else -0.0002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right", fontsize=7
        )

    # Plot B: Class delta distribution
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(df_best["delta_acc"], bins=40, color="#4c9be8", edgecolor="white", linewidth=0.4)
    ax_b.axvline(0, color="#e05c5c", linewidth=1.2, linestyle="--", label="no change")
    ax_b.axvline(df_best["delta_acc"].mean(), color="#f5c842", linewidth=1.2,
                 linestyle="-", label=f"mean={df_best['delta_acc'].mean():+.4f}")
    ax_b.set_xlabel("Δ Accuracy per Class", fontsize=9)
    ax_b.set_ylabel("# Classes", fontsize=9)
    ax_b.set_title(f"B  Δ Acc Distribution\n(nv={best_info['nviews']}, {best_info['strategy']})", fontweight="bold", fontsize=10)
    ax_b.legend(fontsize=7)

    # Plot C: Top/bottom N classes
    ax_c = fig.add_subplot(gs[0, 2])
    N = 20
    combined = pd.concat([df_best.head(N), df_best.tail(N)]).drop_duplicates("class_id").sort_values("delta_acc")
    bar_colors = ["#e05c5c" if v < 0 else "#4c9be8" for v in combined["delta_acc"]]
    ax_c.barh(combined["class_id"].astype(str), combined["delta_acc"], color=bar_colors, edgecolor="white", height=0.7)
    ax_c.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_c.set_xlabel("Δ Accuracy", fontsize=9)
    ax_c.set_title(f"C  Top/Bottom {N} Classes", fontweight="bold", fontsize=10)
    ax_c.tick_params(axis="y", labelsize=6)

    # Plot D: Views scaling curve
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.plot(views_summary["nviews"], views_summary["mean_delta_acc"], marker="o", color="#4c9be8", linewidth=2, markersize=6)
    ax_d.fill_between(
        views_summary["nviews"],
        views_summary["mean_delta_acc"] - views_summary["std_delta_acc"],
        views_summary["mean_delta_acc"] + views_summary["std_delta_acc"],
        alpha=0.15, color="#4c9be8"
    )
    ax_d.axhline(0, color="#e05c5c", linewidth=0.8, linestyle="--")
    ax_d.set_xlabel("N Views", fontsize=9)
    ax_d.set_ylabel("Mean Δ Accuracy", fontsize=9)
    ax_d.set_title(f"D  Views Scaling ({best_info['strategy']})", fontweight="bold", fontsize=10)

    # Plot E: Baseline acc vs delta
    ax_e = fig.add_subplot(gs[1, 1])
    sc = ax_e.scatter(df_best["base_acc"], df_best["delta_acc"], c=df_best["delta_acc"], cmap="RdYlGn", s=12, alpha=0.6, vmin=-0.5, vmax=0.5)
    ax_e.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_e.set_xlabel("Baseline Accuracy", fontsize=9)
    ax_e.set_ylabel("Δ Accuracy (TTA − Baseline)", fontsize=9)
    ax_e.set_title("E  Baseline vs Δ Accuracy", fontweight="bold", fontsize=10)
    fig.colorbar(sc, ax=ax_e, label="Δ Acc", fraction=0.04)

    # Plot F: Delta entropy vs delta accuracy
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.scatter(df_best["delta_ent"], df_best["delta_acc"], c=df_best["base_acc"], cmap="viridis", s=12, alpha=0.6)
    ax_f.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_f.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)
    rho, p = spearmanr(df_best["delta_ent"], df_best["delta_acc"], nan_policy="omit")
    ax_f.set_xlabel("Δ Entropy (↓ = less confused)", fontsize=9)
    ax_f.set_ylabel("Δ Accuracy", fontsize=9)
    ax_f.set_title(f"F  Entropy vs Accuracy Delta\nρ={rho:+.3f}  p={p:.4f}", fontweight="bold", fontsize=10)

    # Apply global styling
    for ax in fig.axes:
        ax.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
    fig.patch.set_facecolor("#12121f")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved summary plot to: {output_path}")
    plt.close(fig)



# ── Main Orchestration Pipeline ────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1 & 2. Load data and clean
    df, _ = load_and_merge_data(BASELINE_PATH, TTA_DIR, CLASSIFIER)

    # 3. Process rankings and determine best config
    config_summary, best_info, views_summary = generate_summaries(df)

    print("\n── Config Ranking (by mean Δ accuracy) ──────────────────────────────")
    print(config_summary.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n✅ Best config: nviews={best_info['nviews']}, eval_strategy={best_info['strategy']} "
          f"(mean Δacc={best_info['mean_delta_acc']:+.4f})")

    # Filter to best data frame
    df_best = df[(df["nviews"] == best_info["nviews"]) & (df["eval_strategy"] == best_info["strategy"])].copy()
    df_best = df_best.sort_values("delta_acc", ascending=False).reset_index(drop=True)

    # Save CSV targets
    config_summary.to_csv(os.path.join(OUTPUT_DIR, "config_ranking.csv"), index=False)
    df_best.to_csv(os.path.join(OUTPUT_DIR, "class_deltas_best_config.csv"), index=False)
    views_summary.to_csv(os.path.join(OUTPUT_DIR, "views_scaling_summary.csv"), index=False)

    # 4. Class-Level Terminal Printing
    print(f"\n── Class-level (best config) ─────────────────────────────────────────")
    print(f"  Improved : {(df_best['delta_acc'] > 0).sum()}  |  Hurt : {(df_best['delta_acc'] < 0).sum()}  |  Neutral : {(df_best['delta_acc'] == 0).sum()}")
    print(f"  Top 5 improved classes:\n{df_best.head(5)[['class_id','base_acc','tta_acc','delta_acc']].to_string(index=False)}")
    print(f"  Top 5 hurt classes:\n{df_best.tail(5)[['class_id','base_acc','tta_acc','delta_acc']].to_string(index=False)}")

    # 5. Views Scaling Terminal Printing
    print(f"\n── Views Scaling (eval_strategy={best_info['strategy']}) ─────────────────────")
    print(views_summary.to_string(index=False, float_format="{:.4f}".format))

    # 6. Domain Stats Hook Hook Execution
    run_domain_hook_analysis(df_best, json_path="path/to/domain_stats.json")

    # 7. Visualization
    plot_path = os.path.join(OUTPUT_DIR, "geo_tta_analysis.png")
    plot_tta_analysis(CLASSIFIER, config_summary, df_best, views_summary, best_info, plot_path)


if __name__ == "__main__":
    main()
