import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from code.config.paths import OUTPUT_PATH

ALL_CLASSIFIERS = ["densenet121", "swin_base_patch4_window7_224", "dinov2_vitb14"]
ALL_SEEDS = [71397589]
DATASET_SPLITS = {
    "imagenet": "test_r",
    "eurosat": "ucmerced",
    "midog": "test",
    "camelyon17wilds": "test",
    "epistr": "test",
}
MODEL_DISPLAY_NAMES = {
    "densenet121": "CNN",
    "swin_base_patch4_window7_224": "ViT",
    "dinov2_vitb14": "FM",
}

def load_view_predictions(path: Path):
    with open(path, "r") as f:
        results = json.load(f)
    y_true = np.array([s["y_true"] for s in results["predictions"]])
    y_prob = np.array([s["y_pred"] for s in results["predictions"]])
    return y_true, y_prob


def collect_model_entropy_data(base_dir: Path, ds: str, split: str, cl: str, seed: int, n_views: int = 15):
    all_probs = []
    y_true = None

    for v_idx in range(1, n_views + 1):
        view_str = f"view_{v_idx:03d}.png" if ds == "imagenet" else f"view_{v_idx:03d}"
        fname = f"{cl}_geometric_vanilla_nviews1_seed{seed}.json"
        pred_path = base_dir / "style_check" / "styleid" / view_str / "tta_inference/predictions" / ds / split / fname

        if not pred_path.exists():
            return None, None

        yt, prob = load_view_predictions(pred_path)
        if y_true is None:
            y_true = yt
        all_probs.append(prob)

    probs_stack = np.stack(all_probs, axis=0)
    n_v, n_s, n_c = probs_stack.shape

    flat_probs = probs_stack.reshape(-1, n_c)
    flat_y_true = np.tile(y_true, n_v)

    flat_preds = np.argmax(flat_probs, axis=-1)
    correctness = (flat_preds == flat_y_true).astype(float)

    eps = 1e-12
    entropies = -np.sum(flat_probs * np.log(flat_probs + eps), axis=-1)

    return entropies, correctness


def generate_accuracy_vs_entropy_plots(results_dir: Path, output_dir: Path, n_bins: int = 15):
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds, split in DATASET_SPLITS.items():
        print(f"Processing dataset: {ds.upper()}")
        results_per_model = {}

        for cl in ALL_CLASSIFIERS:
            all_entropies, all_correctness = [], []

            for seed in ALL_SEEDS:
                entropies, correctness = collect_model_entropy_data(results_dir, ds, split, cl, seed)
                if entropies is not None:
                    all_entropies.append(entropies)
                    all_correctness.append(correctness)

            if not all_entropies:
                print(f"  [{cl}] Missing files, skipping.")
                continue

            concat_entropies = np.concatenate(all_entropies)
            concat_correctness = np.concatenate(all_correctness)
            results_per_model[cl] = (concat_entropies, concat_correctness)

        if not results_per_model:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name, (entropies, correctness) in results_per_model.items():
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            quantiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(entropies, quantiles)

            bin_accs, bin_ents = [], []
            for i in range(n_bins):
                low, high = bin_edges[i], bin_edges[i + 1]
                if i == n_bins - 1:
                    mask = (entropies >= low) & (entropies <= high)
                else:
                    mask = (entropies >= low) & (entropies < high)

                if np.sum(mask) > 0:
                    bin_accs.append(np.mean(correctness[mask]) * 100)
                    bin_ents.append(np.mean(entropies[mask]))

            ax.plot(bin_ents, bin_accs, marker="o", linewidth=2, label=display_name)

        ax.set_xlabel("Predictive Entropy (nats)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Accuracy vs. Entropy across Stylized Views ({ds.upper()})")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()

        out_plot_path = output_dir / f"accuracy_vs_entropy_{ds}.png"
        plt.savefig(out_plot_path, dpi=300)
        plt.close()
        print(f"  Saved plot to {out_plot_path}")


if __name__ == "__main__":
    results_base = OUTPUT_PATH
    output_base = Path("./output/entropy_plots")
    generate_accuracy_vs_entropy_plots(results_base, output_base)
