import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from code.config.constants import (
    ALL_CLASSIFIERS,
    ALL_SEEDS,
    TTA_STRATEGIES,
)


def get_prediction_filename(s_key, cfg, ds, cl, ev, retr, rfs, seed, sty=1, use_n=1):
    target_cfg = TTA_STRATEGIES.get(s_key, cfg)

    if s_key == "hybrid_tta":
        geo = (rfs - 1) - sty
        return "{ds}_{cl}_hybrid_geo{geo}_sty{sty}_{eval}_split{use_n}_nr{rfs}_seed{seed}_predictions.json".format(
            ds=ds,
            cl=cl,
            geo=f"{geo:02d}",
            sty=f"{sty:02d}",
            eval=ev,
            use_n=use_n,
            rfs=rfs,
            seed=seed,
        )

    fmt_kwargs = {
        "ds": ds,
        "cl": cl,
        "eval": ev if ev else target_cfg.get("default_eval", "vanilla"),
        "rfs": rfs,
        "seed": seed,
        "retr": retr if retr is not None else target_cfg.get("default_retr", ""),
    }

    try:
        return target_cfg["template"].format(**fmt_kwargs)
    except KeyError:
        return target_cfg["template"].format(cl=cl, rfs=rfs, seed=seed)


def load_prediction_probabilities(path: Path):
    with open(path, "r") as f:
        results = json.load(f)
    y_true = np.array([sample["y_true"] for sample in results["predictions"]])
    y_prob = np.array([sample["y_pred"] for sample in results["predictions"]])
    return y_true, y_prob


def compute_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            ece += (bin_count / n_samples) * np.abs(bin_acc - bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2)

    return {
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_boundaries": bin_boundaries,
        "ece": float(ece),
    }


def _draw_overlaid_reliability(
    ax,
    y_true_a: np.ndarray,
    y_prob_a: np.ndarray,
    label_a: str,
    y_true_b: np.ndarray,
    y_prob_b: np.ndarray,
    label_b: str,
    n_bins: int = 15,
):
    cal_a = compute_calibration_bins(y_true_a, y_prob_a, n_bins=n_bins)
    cal_b = compute_calibration_bins(y_true_b, y_prob_b, n_bins=n_bins)

    bin_centers = (cal_a["bin_boundaries"][:-1] + cal_a["bin_boundaries"][1:]) / 2
    bin_width = (1.0 / n_bins) * 0.4

    ax.bar(
        bin_centers - bin_width / 2,
        cal_a["bin_accs"],
        width=bin_width,
        edgecolor="black",
        color="#1f77b4",
        alpha=0.8,
        label=f"{label_a} (ECE: {cal_a['ece']:.4f})",
    )
    ax.bar(
        bin_centers + bin_width / 2,
        cal_b["bin_accs"],
        width=bin_width,
        edgecolor="black",
        color="#2ca02c",
        alpha=0.8,
        label=f"{label_b} (ECE: {cal_b['ece']:.4f})",
    )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    return cal_a["ece"], cal_b["ece"]


def plot_reliability_comparison_multi_seed(
    seed_paths: Dict[int, Tuple[Path, Path]],
    output_plot_path: Path,
    output_json_path: Path,
    n_bins: int = 15,
    title: str = "Reliability Comparison",
):
    all_yt_base, all_prob_base = [], []
    all_yt_tta, all_prob_tta = [], []
    seed_eces_base = {}
    seed_eces_tta = {}

    for s_id, (base_path, tta_path) in seed_paths.items():
        yt_b, prob_b = load_prediction_probabilities(base_path)
        yt_t, prob_t = load_prediction_probabilities(tta_path)

        all_yt_base.append(yt_b)
        all_prob_base.append(prob_b)
        all_yt_tta.append(yt_t)
        all_prob_tta.append(prob_t)

        seed_eces_base[s_id] = compute_calibration_bins(yt_b, prob_b, n_bins=n_bins)["ece"]
        seed_eces_tta[s_id] = compute_calibration_bins(yt_t, prob_t, n_bins=n_bins)["ece"]

    pooled_yt_base = np.concatenate(all_yt_base)
    pooled_prob_base = np.concatenate(all_prob_base)
    pooled_yt_tta = np.concatenate(all_yt_tta)
    pooled_prob_tta = np.concatenate(all_prob_tta)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ece_base_pooled, ece_tta_pooled = _draw_overlaid_reliability(
        ax,
        pooled_yt_base,
        pooled_prob_base,
        "Geometric TTA (rfs=64)",
        pooled_yt_tta,
        pooled_prob_tta,
        "STyle-TTA (rfs=16)",
        n_bins=n_bins,
    )

    #ax.set_title(title, fontsize=12)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    metrics = {
        "ece_geometric64_pooled": ece_base_pooled,
        "ece_style_tta16_pooled": ece_tta_pooled,
        "ece_geometric64_seeds_mean": float(np.mean(list(seed_eces_base.values()))),
        "ece_style_tta16_seeds_mean": float(np.mean(list(seed_eces_tta.values()))),
        "seed_eces_geometric64": seed_eces_base,
        "seed_eces_style_tta16": seed_eces_tta,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_all_reliability_evaluations(
    results_dir: Path,
    output_dir: Path,
    datasets: list,
    n_bins: int = 15,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_base = "geometric_tta"
    cfg_base = TTA_STRATEGIES[strategy_base]
    rfs_base = 64

    strategy_target = "ablation/style_tta"
    cfg_target = TTA_STRATEGIES[strategy_target]
    rfs_target = 16

    for ds in datasets:
        print(f"\n=== Starting dataset: {ds} ===")

        split = "test"
        if ds == "eurosat":
            split = "ucmerced"
        elif ds == "imagenet":
            split = "test_r"

        ev_base = cfg_base.get("default_eval", "vanilla")
        retr_base = cfg_base.get("default_retr", "")

        ev_target = cfg_target.get("default_eval", "vanilla")
        retr_target = cfg_target.get("default_retr", "")

        for cl in ALL_CLASSIFIERS:
            seed_paths = {}

            for seed in ALL_SEEDS:
                base_fname = get_prediction_filename(
                    strategy_base, cfg_base, ds, cl, ev_base, retr_base,
                    rfs=rfs_base, seed=seed
                )
                base_pred_path = results_dir / strategy_base / f"tta_inference/predictions/{ds}/{split}" / base_fname

                target_fname = get_prediction_filename(
                    strategy_target, cfg_target, ds, cl, ev_target, retr_target,
                    rfs=rfs_target, seed=seed
                )
                target_pred_path = results_dir / strategy_target / f"tta_inference/predictions/{ds}/{split}" / target_fname

                if not base_pred_path.exists() or not target_pred_path.exists():
                    continue

                seed_paths[seed] = (base_pred_path, target_pred_path)

            if not seed_paths:
                continue

            out_plot = output_dir / "geom64_vs_style_tta16" / ds / cl / "reliability_comparison.png"
            out_json = output_dir / "geom64_vs_style_tta16" / ds / cl / "ece_comparison.json"
            title_str = f"{ds.upper()} | {cl} | Geometric (rfs=64) vs Retristyle (rfs=16)"

            plot_reliability_comparison_multi_seed(
                seed_paths=seed_paths,
                output_plot_path=out_plot,
                output_json_path=out_json,
                n_bins=n_bins,
                title=title_str,
            )

        print(f"=== Finished dataset: {ds} ===\n")


if __name__ == "__main__":
    print("RELIABILITY DIAGRAM SCRIPT STARTED", flush=True)
    run_all_reliability_evaluations(
        results_dir=Path("./results"),
        output_dir=Path("./output/reliability_plots"),
        datasets=["imagenet"],
        n_bins=15,
    )
