import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ALL_CLASSIFIERS = {
    "densenet121": "CNN",
    "swin_base_patch4_window7_224": "ViT",
    "dinov2_vitb14": "FM",
}
ALL_SEEDS = [71397589]
DATASET_SPLITS = {
    "imagenet": "test_r",
    "eurosat": "ucmerced",
    "midog": "test",
    "camelyon17wilds": "test",
    "epistr": "test",
}


def load_predictions(path: Path):
    with open(path, "r") as f:
        results = json.load(f)
    y_true = np.array([s["y_true"] for s in results["predictions"]])
    y_prob = np.array([s["y_pred"] for s in results["predictions"]])
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


def get_baseline_predictions(base_dir: Path, ds: str, split: str, cl: str, seed: int):
    fname = f"{cl}_geometric_vanilla_nviews1_seed{seed}.json"
    path = base_dir / "geometric_tta" / "tta_inference" / "predictions" / ds / split / fname
    if not path.exists():
        return None, None, None
    return load_predictions(path)


def analyze_view_order_and_gain(base_dir: Path, ds: str, split: str, cl: str, seed: int, max_views: int = 15):
    y_true, y_pred_base, _ = get_baseline_predictions(base_dir, ds, split, cl, seed)
    if y_true is None:
        return None

    base_correct = (y_pred_base == y_true).astype(float)

    all_probs = []
    single_view_accs = []
    single_view_gains = []

    for v_idx in range(1, max_views + 1):
        view_str = f"view_{v_idx:03d}.png" if ds == "imagenet" else f"view_{v_idx:03d}"
        fname = f"{cl}_geometric_vanilla_nviews1_seed{seed}.json"
        pred_path = base_dir / "style_check" / "styleid" / view_str / "tta_inference/predictions" / ds / split / fname

        if not pred_path.exists():
            return None

        yt, yp, prob = load_predictions(pred_path)
        view_correct = (yp == yt).astype(float)
        
        all_probs.append(prob)
        single_view_accs.append(np.mean(view_correct) * 100)
        single_view_gains.append(np.mean(view_correct - base_correct) * 100)

    probs_stack = np.stack(all_probs, axis=0)  # (n_views, n_samples, n_classes)

    marginal_accs = []
    marginal_gains = []

    for k in range(1, max_views + 1):
        cum_probs = np.mean(probs_stack[:k], axis=0)
        cum_preds = np.argmax(cum_probs, axis=1)
        cum_correct = (cum_preds == y_true).astype(float)
        
        marginal_accs.append(np.mean(cum_correct) * 100)
        marginal_gains.append(np.mean(cum_correct - base_correct) * 100)

    return {
        "single_view_accs": single_view_accs,
        "single_view_gains": single_view_gains,
        "marginal_accs": marginal_accs,
        "marginal_gains": marginal_gains,
    }


def run_view_rank_analysis(results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds, split in DATASET_SPLITS.items():
        print(f"\n================ DATASET: {ds.upper()} ================")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for cl_raw, cl_display in ALL_CLASSIFIERS.items():
            single_accs_seeds = []
            marginal_accs_seeds = []

            for seed in ALL_SEEDS:
                res = analyze_view_order_and_gain(results_dir, ds, split, cl_raw, seed)
                if res is not None:
                    single_accs_seeds.append(res["single_view_accs"])
                    marginal_accs_seeds.append(res["marginal_accs"])

            if not single_accs_seeds:
                print(f"[{cl_display}] Missing prediction files.")
                continue

            mean_single_accs = np.mean(single_accs_seeds, axis=0)
            mean_marginal_accs = np.mean(marginal_accs_seeds, axis=0)
            views_range = np.arange(1, len(mean_single_accs) + 1)

            print(f"\nModel: {cl_display} ({cl_raw})")
            print("  Rank  | Standalone Acc (%) | Cumulative Ensemble Acc (%)")
            print("  -------------------------------------------------------")
            for r in range(max(5, len(mean_single_accs))):
                print(f"  View {r+1:02d} | {mean_single_accs[r]:18.2f} | {mean_marginal_accs[r]:25.2f}")

            axes[0].plot(views_range, mean_single_accs, marker="o", linewidth=2, label=cl_display)
            axes[1].plot(views_range, mean_marginal_accs, marker="s", linewidth=2, label=cl_display)

        axes[0].set_xlabel("Retrieval Rank (View Index)")
        axes[0].set_ylabel("Standalone Accuracy (%)")
        axes[0].set_title(f"Individual View Quality by Rank ({ds.upper()})")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        axes[1].set_xlabel("Number of Ensembled Views (Top-K)")
        axes[1].set_ylabel("Ensemble Accuracy (%)")
        axes[1].set_title(f"Cumulative Ensemble Scaling ({ds.upper()})")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        out_plot_path = output_dir / f"view_rank_analysis_{ds}.png"
        plt.savefig(out_plot_path, dpi=300)
        plt.close()
        print(f"\nSaved plot to {out_plot_path}")


if __name__ == "__main__":
    results_base = Path("/home/stud/nemmler/retristyle/results")
    output_base = Path("./output/view_rank_plots")
    run_view_rank_analysis(results_base, output_base)
