import json
from code.config.paths import OUTPUT_PATH
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr

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
        return None, None
    yt, yp, prob = load_predictions(path)
    return yt, yp


def compute_pairwise_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


def compute_sample_view_diversity(probs_stack: np.ndarray):
    # probs_stack shape: (n_views, n_samples, n_classes)
    n_views, n_samples, _ = probs_stack.shape
    
    mean_kl = np.zeros(n_samples)
    mean_cosine = np.zeros(n_samples)
    
    n_pairs = n_views * (n_views - 1) // 2

    for i in range(n_samples):
        kl_sum = 0.0
        cos_sum = 0.0
        for v1 in range(n_views):
            for v2 in range(v1 + 1, n_views):
                p1, p2 = probs_stack[v1, i], probs_stack[v2, i]
                # Symmetric KL divergence
                kl_sum += 0.5 * (compute_pairwise_kl_divergence(p1, p2) + compute_pairwise_kl_divergence(p2, p1))
                cos_sum += cosine(p1, p2)
                
        mean_kl[i] = kl_sum / n_pairs
        mean_cosine[i] = cos_sum / n_pairs

    return mean_kl, mean_cosine


def compute_diversity_and_correlation(base_dir: Path, ds: str, split: str, cl: str, seed: int, n_views: int = 15):
    y_true, y_pred_base = get_baseline_predictions(base_dir, ds, split, cl, seed)
    if y_true is None:
        return None

    all_preds, all_probs = [], []
    for v_idx in range(1, n_views + 1):
        view_str = f"view_{v_idx:03d}.png" if ds == "imagenet" else f"view_{v_idx:03d}"
        fname = f"{cl}_geometric_vanilla_nviews1_seed{seed}.json"
        pred_path = base_dir / "style_check" / "styleid" / view_str / "tta_inference/predictions" / ds / split / fname

        if not pred_path.exists():
            return None
        _, yp, prob = load_predictions(pred_path)
        all_preds.append(yp)
        all_probs.append(prob)

    preds_matrix = np.column_stack(all_preds)       # (n_samples, n_views)
    probs_stack = np.stack(all_probs, axis=0)        # (n_views, n_samples, n_classes)

    # 1. Full Agreement Across All 15 Views
    all_agree = np.all(preds_matrix == preds_matrix[:, [0]], axis=1)
    full_agreement_pct = np.mean(all_agree) * 100

    # 2. Pairwise Softmax Diversity (KL and Cosine)
    sample_kl, sample_cosine = compute_sample_view_diversity(probs_stack)

    # 3. Per-sample Accuracy Gain (Ensemble Correctness - Baseline Correctness)
    ensemble_probs = np.mean(probs_stack, axis=0)
    y_pred_ens = np.argmax(ensemble_probs, axis=1)
    
    base_correct = (y_pred_base == y_true).astype(float)
    ens_correct = (y_pred_ens == y_true).astype(float)
    acc_gain = ens_correct - base_correct

    # 4. Correlations between Diversity and Accuracy Gain
    p_corr_kl, _ = pearsonr(sample_kl, acc_gain)
    s_corr_kl, _ = spearmanr(sample_kl, acc_gain)
    
    p_corr_cos, _ = pearsonr(sample_cosine, acc_gain)
    s_corr_cos, _ = spearmanr(sample_cosine, acc_gain)

    return {
        "full_agreement_pct": full_agreement_pct,
        "mean_kl": np.mean(sample_kl),
        "mean_cosine": np.mean(sample_cosine),
        "pearson_kl": p_corr_kl,
        "spearman_kl": s_corr_kl,
        "pearson_cos": p_corr_cos,
        "spearman_cos": s_corr_cos,
    }


def run_diversity_analysis(results_dir: Path):
    for ds, split in DATASET_SPLITS.items():
        print(f"\n================ DATASET: {ds.upper()} ================")
        for cl_raw, cl_display in ALL_CLASSIFIERS.items():
            metrics_list = []

            for seed in ALL_SEEDS:
                res = compute_diversity_and_correlation(results_dir, ds, split, cl_raw, seed)
                if res is not None:
                    metrics_list.append(res)

            if not metrics_list:
                print(f"[{cl_display}] Missing prediction files.")
                continue

            agr = np.mean([m["full_agreement_pct"] for m in metrics_list])
            kl = np.mean([m["mean_kl"] for m in metrics_list])
            cos = np.mean([m["mean_cosine"] for m in metrics_list])
            p_kl = np.mean([m["pearson_kl"] for m in metrics_list])
            s_kl = np.mean([m["spearman_kl"] for m in metrics_list])
            p_cos = np.mean([m["pearson_cos"] for m in metrics_list])
            s_cos = np.mean([m["spearman_cos"] for m in metrics_list])

            print(f"\nModel: {cl_display} ({cl_raw})")
            print(f"  15-View Full Agreement: {agr:.2f}%")
            print(f"  Avg Pairwise Sym-KL  : {kl:.4f}")
            print(f"  Avg Pairwise Cosine  : {cos:.4f}")
            print(f"  Corr(KL, Acc Gain)   : Pearson = {p_kl:+.3f} | Spearman = {s_kl:+.3f}")
            print(f"  Corr(Cos, Acc Gain)  : Pearson = {p_cos:+.3f} | Spearman = {s_cos:+.3f}")


if __name__ == "__main__":
    results_base = OUTPUT_PATH
    run_diversity_analysis(results_base)
