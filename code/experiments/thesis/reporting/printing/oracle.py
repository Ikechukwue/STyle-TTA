import json
from pathlib import Path
import numpy as np

ALL_CLASSIFIERS = ["densenet121", "swin_base_patch4_window7_224", "dinov2_vitb14"]
ALL_SEEDS = [71397589]
DATASET_SPLITS = {
    "imagenet": "test_r",
    "eurosat": "ucmerced",
    "midog": "test",
    "camelyon17wilds": "test",
    "epistr": "test",
}


def load_view_predictions(path: Path):
    with open(path, "r") as f:
        results = json.load(f)
    y_true = np.array([s["y_true"] for s in results["predictions"]])
    y_prob = np.array([s["y_pred"] for s in results["predictions"]])
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob

ENSEMBLE_SIZES = [2, 4, 8, 16]


def compute_oracle_and_ensemble_scaling(base_dir: Path, ds: str, split: str, cl: str, seed: int, max_views: int = 15):
    all_preds, all_probs = [], []
    y_true = None

    for v_idx in range(1, max_views + 1):
        view_str = f"view_{v_idx:03d}.png" if ds == "imagenet" else f"view_{v_idx:03d}"
        fname = f"{cl}_geometric_vanilla_nviews1_seed{seed}.json"
        pred_path = base_dir / "style_check" / "styleid" / view_str / "tta_inference/predictions" / ds / split / fname

        if not pred_path.exists():
            return None

        yt, yp, prob = load_view_predictions(pred_path)
        if y_true is None:
            y_true = yt

        all_preds.append(yp)
        all_probs.append(prob)

    preds_matrix = np.column_stack(all_preds)
    probs_stack = np.stack(all_probs, axis=0)

    results_by_k = {}
    for k in ENSEMBLE_SIZES:
        actual_k = min(k, max_views)
        
        sub_preds = preds_matrix[:, :actual_k]
        oracle_hits = np.any(sub_preds == y_true[:, None], axis=1)
        oracle_acc = float(np.mean(oracle_hits))

        sub_probs = probs_stack[:actual_k]
        mean_probs = np.mean(sub_probs, axis=0)
        ensemble_preds = np.argmax(mean_probs, axis=1)
        ensemble_acc = float(np.mean(ensemble_preds == y_true))

        results_by_k[k] = (oracle_acc, ensemble_acc)

    return results_by_k


def run_oracle_evaluation(results_dir: Path):
    for ds, split in DATASET_SPLITS.items():
        print(f"\n================ DATASET: {ds.upper()} ================")
        for cl in ALL_CLASSIFIERS:
            scaling_metrics = {k: {"oracle": [], "ensemble": []} for k in ENSEMBLE_SIZES}

            for seed in ALL_SEEDS:
                max_int = 15 if ds == "imagenet" else 4
                metrics_by_k = compute_oracle_and_ensemble_scaling(results_dir, ds, split, cl, seed)
                if metrics_by_k is None:
                    continue
                for k, (o_acc, e_acc) in metrics_by_k.items():
                    scaling_metrics[k]["oracle"].append(o_acc)
                    scaling_metrics[k]["ensemble"].append(e_acc)

            if not scaling_metrics[ENSEMBLE_SIZES[0]]["oracle"]:
                print(f"[{cl}] Missing prediction files.")
                continue

            print(f"\nClassifier: {cl}")
            for k in ENSEMBLE_SIZES:
                m_oracle = np.mean(scaling_metrics[k]["oracle"]) * 100
                m_ensemble = np.mean(scaling_metrics[k]["ensemble"]) * 100
                gap = m_oracle - m_ensemble
                k_label = f"Top-{k}" if k <= 15 else "Top-16 (15)"
                print(f"  {k_label:<12} | Ens Acc: {m_ensemble:.2f}% | Oracle Upper Bound: {m_oracle:.2f}% | Gap: +{gap:.2f}%")


if __name__ == "__main__":
    results_base = Path("/home/stud/nemmler/retristyle/results")
    run_oracle_evaluation(results_base)
