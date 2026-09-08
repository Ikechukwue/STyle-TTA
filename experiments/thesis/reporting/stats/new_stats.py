import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config.constants import (
    ALL_CLASSIFIERS,
    TTA_STRATEGIES,
    DEFAULT_SEED,
    ALL_DATASETS,
    RETRIEVAL_STRATEGIES,
    ALL_SEEDS,
)
from config.helpers import baseline_results


# ==============================================================================
# Helper Functions: Filename Resolution & Prediction Loading
# ==============================================================================

def get_prediction_filename(
    s_key: str,
    cfg: Dict[str, Any],
    ds: str,
    cl: str,
    ev: str,
    retr: str,
    rfs: int,
    seed: int,
    sty: int = 1,
    use_n: int = 1,
) -> str:
    target_cfg = TTA_STRATEGIES.get(s_key, cfg)

    if s_key == "hybrid_tta":
        geo = (rfs - 1) - sty
        return f"{ds}_{cl}_hybrid_geo{geo:02d}_sty{sty:02d}_{ev}_split{use_n}_nr{rfs}_seed{seed}_predictions.json"

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


def get_predictions_and_probs(results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts y_true, discrete predicted classes (y_pred), and 
    predicted probability distributions (y_prob).
    """
    y_true = np.array([sample["y_true"] for sample in results["predictions"]])
    y_prob = np.array([sample["y_pred"] for sample in results["predictions"]], dtype=np.float64)

    # Convert raw logits to probabilities via softmax if unnormalized
    if not np.allclose(np.sum(y_prob, axis=1), 1.0, atol=1e-3):
        exp_p = np.exp(y_prob - np.max(y_prob, axis=1, keepdims=True))
        y_prob = exp_p / np.sum(exp_p, axis=1, keepdims=True)

    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob


# ==============================================================================
# Metric Computation Routines
# ==============================================================================

def calculate_ece_np(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) over predicted probabilities."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes Accuracy, Balanced Accuracy, Macro/Binary AUC, and ECE."""
    acc = float(np.mean(y_true == y_pred))

    classes = np.unique(y_true)
    recalls = [np.mean(y_pred[y_true == cls] == cls) for cls in classes if np.sum(y_true == cls) > 0]
    bal_acc = float(np.mean(recalls))

    try:
        if len(classes) == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    ece = calculate_ece_np(y_true, y_prob)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc": auc,
        "ece": ece,
    }


# ==============================================================================
# Statistical Testing & Bootstrapping
# ==============================================================================

def stratified_bootstrap_all_metrics(
    y_true: np.ndarray,
    pred_base: np.ndarray,
    prob_base: np.ndarray,
    pred_tta: np.ndarray,
    prob_tta: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Performs stratified bootstrapping to compute 95% CIs and two-sided bootstrap p-values
    for the metric differences (TTA - Baseline) across all evaluated metrics.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_true)
    class_indices = {cls: np.where(y_true == cls)[0] for cls in classes}

    diff_store = {
        "accuracy": np.empty(n_bootstrap),
        "balanced_accuracy": np.empty(n_bootstrap),
        "auc": np.empty(n_bootstrap),
        "ece": np.empty(n_bootstrap),
    }

    for b in range(n_bootstrap):
        sampled_indices = np.concatenate([
            rng.choice(class_indices[cls], size=len(class_indices[cls]), replace=True)
            for cls in classes
        ])

        y_true_b = y_true[sampled_indices]
        m_base_b = compute_all_metrics(y_true_b, pred_base[sampled_indices], prob_base[sampled_indices])
        m_tta_b = compute_all_metrics(y_true_b, pred_tta[sampled_indices], prob_tta[sampled_indices])

        for m_key in diff_store:
            diff_store[m_key][b] = m_tta_b[m_key] - m_base_b[m_key]

    alpha = 1.0 - confidence_level
    observed_base = compute_all_metrics(y_true, pred_base, prob_base)
    observed_tta = compute_all_metrics(y_true, pred_tta, prob_tta)

    output = {}
    for m_key in diff_store:
        diffs = diff_store[m_key]
        
        # Empirical two-sided bootstrap p-value null-hypothesis test
        p_val = float(np.mean(np.abs(diffs) >= np.abs(np.mean(diffs))))

        output[m_key] = {
            "baseline": observed_base[m_key],
            "tta": observed_tta[m_key],
            "difference": float(observed_tta[m_key] - observed_base[m_key]),
            "ci_lower": float(np.percentile(diffs, 100 * (alpha / 2))),
            "ci_upper": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
            "bootstrap_p_value": p_val,
        }

    return output


def calculate_statistical_comparison(
    baseline_path: Path,
    tta_path: Path,
    output_path: Path,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calculates statistics for all metrics and dumps output JSON."""
    with open(baseline_path, "r") as f:
        baseline_results = json.load(f)

    with open(tta_path, "r") as f:
        tta_results = json.load(f)

    y_true_base, pred_base, prob_base = get_predictions_and_probs(baseline_results)
    y_true_tta, pred_tta, prob_tta = get_predictions_and_probs(tta_results)

    if len(y_true_base) != len(y_true_tta):
        raise ValueError(f"Baseline ({len(y_true_base)}) and TTA ({len(y_true_tta)}) sample counts mismatch.")

    if not np.array_equal(y_true_base, y_true_tta):
        raise ValueError("Ground-truth labels do not match between baseline and TTA.")

    # McNemar's exact test for classification error concordancy
    b = np.sum((pred_base == y_true_base) & (pred_tta != y_true_base))
    c = np.sum((pred_base != y_true_base) & (pred_tta == y_true_base))
    mcnemar_p = float(binomtest(k=min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue) if (b + c) > 0 else 1.0

    metrics_stats = stratified_bootstrap_all_metrics(
        y_true=y_true_base,
        pred_base=pred_base, prob_base=prob_base,
        pred_tta=pred_tta, prob_tta=prob_tta,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )

    result = {
        "baseline_file": str(baseline_path),
        "tta_file": str(tta_path),
        "n_samples": int(len(y_true_base)),
        "n_classes": int(len(np.unique(y_true_base))),
        "mcnemar": {
            "baseline_correct_tta_wrong": int(b),
            "baseline_wrong_tta_correct": int(c),
            "discordant_pairs": int(b + c),
            "p_value": mcnemar_p,
        },
        "metrics": metrics_stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ==============================================================================
# Pipeline Execution Functions
# ==============================================================================

def set_against_path(
    against: str,
    cfg: Dict[str, Any],
    ds: str,
    cl: str,
    seed: int,
    results_dir: Path,
    split: str,
    s_key: str,
    rfs: int = 1,
    sty: int = 3,
    use_n: int = 4,
) -> Optional[Path]:
    if against == "baseline":
        base_fname = get_prediction_filename("geometric_tta", cfg, ds, cl, "vanilla", "", rfs=1, seed=seed)
        base_pred_path = results_dir / f"geometric_tta/tta_inference/predictions/{ds}/{split}" / base_fname
        
        if not base_pred_path.exists():
            base_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}" / base_fname
            if not base_pred_path.exists():
                return None
    else:
        ev = "zero" if against == "adain_tta" else "vanilla"
        retr = "" if against == "geometric_tta" else "dino"
        base_fname = get_prediction_filename(against, cfg, ds, cl, ev, retr, rfs, seed=seed, sty=sty, use_n=use_n)
        base_pred_path = results_dir / against / f"tta_inference/predictions/{ds}/{split}" / base_fname
    
    return base_pred_path


def run_all_statistical_comparisons(
    results_dir: Path,
    output_dir: Path,
    datasets: List[str],
    against: str = "baseline",
    n_bootstrap: int = 10_000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary_stats: Dict[str, Any] = {}

    for ds in datasets:
        all_summary_stats[ds] = {}
        split = "test"
        if ds == "eurosat":
            split = "ucmerced"
        elif ds == "imagenet":
            split = "test_r"

        for s_key, cfg in TTA_STRATEGIES.items():
            if s_key == against:
                continue
            ev = cfg.get("default_eval", "vanilla")
            retr = cfg.get("default_retr", "")
            all_summary_stats[ds][s_key] = {}
            
            for cl in ALL_CLASSIFIERS:
                all_summary_stats[ds][s_key][cl] = {}
                for seed in ALL_SEEDS:
                    for rfs in cfg["axis"]:
                        if s_key == "hybrid_tta":
                            for sty_val in [1, 2, 3]:
                                for use_n in [1, sty_val + 1]:
                                    tta_fname = get_prediction_filename(
                                        s_key, cfg, ds, cl, ev, retr, rfs=rfs, seed=seed, sty=sty_val, use_n=use_n
                                    )
                                    tta_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}/{split}" / tta_fname
                                    base_pred_path = set_against_path(against, cfg, ds, cl, seed, results_dir, split, s_key, rfs, sty_val, use_n)

                                    if base_pred_path is None or not base_pred_path.exists():
                                        continue

                                    if tta_pred_path.exists():
                                        out_json = output_dir / against / ds / s_key / cl / f"stats_nr{rfs}_sty{sty_val}_split{use_n}_seed{seed}.json"
                                        stats_res = calculate_statistical_comparison(
                                            baseline_path=base_pred_path,
                                            tta_path=tta_pred_path,
                                            output_path=out_json,
                                            n_bootstrap=n_bootstrap,
                                            seed=seed,
                                        )
                                        all_summary_stats[ds][s_key][cl][f"nr_{rfs}_sty_{sty_val}_split_{use_n}_seed_{seed}"] = stats_res
                        else:
                            tta_fname = get_prediction_filename(s_key, cfg, ds, cl, ev, retr, rfs=rfs, seed=seed)
                            tta_pred_path = results_dir / s_key / f"tta_inference/predictions/{ds}/{split}" / tta_fname
                            base_pred_path = set_against_path(against, cfg, ds, cl, seed, results_dir, split, s_key, rfs)

                            if base_pred_path is None or not base_pred_path.exists():
                                continue

                            if tta_pred_path.exists():
                                out_json = output_dir / against / ds / s_key / cl / f"stats_rfs{rfs}_seed{seed}.json"
                                stats_res = calculate_statistical_comparison(
                                    baseline_path=base_pred_path,
                                    tta_path=tta_pred_path,
                                    output_path=out_json,
                                    n_bootstrap=n_bootstrap,
                                    seed=seed,
                                )
                                all_summary_stats[ds][s_key][cl][f"rfs_{rfs}_seed_{seed}"] = stats_res


def run_list_comparisons(
    prediction_pattern: str,
    results_dir: Path,
    output_dir: Path,
    dataset: str,
    split: str,
    n_bootstrap: int = 10_000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary_stats: Dict[str, Any] = {}

    for cls in ALL_CLASSIFIERS:
        for retr in RETRIEVAL_STRATEGIES:
            tta_pred_path = Path(prediction_pattern.format(cl=cls, retr=retr))

            if not tta_pred_path.exists():
                print(f"Skipping non-existent path: {tta_pred_path}")
                continue

            base_fname = f"{cls}_vanilla.json"
            base_pred_path = Path(baseline_results(dataset=dataset, split=split, cls=cls, prediction=True))
            
            if not base_pred_path.exists():
                base_pred_path = results_dir / f"geometric_tta/tta_inference/predictions/{dataset}" / base_fname
                if not base_pred_path.exists():
                    print(f"Baseline not found for {tta_pred_path}, skipping.")
                    continue

            stem = tta_pred_path.stem
            out_json = output_dir / dataset / cls / f"stats_{stem}.json"

            stats_res = calculate_statistical_comparison(
                baseline_path=base_pred_path,
                tta_path=tta_pred_path,
                output_path=out_json,
                n_bootstrap=n_bootstrap,
            )

            all_summary_stats[stem] = stats_res

    with open(output_dir / "aggregated_summary.json", "w") as f:
        json.dump(all_summary_stats, f, indent=2)


if __name__ == "__main__":
    for ag in ["baseline", "geometric_tta"]:
        run_all_statistical_comparisons(
            results_dir=Path("./results"),
            output_dir=Path("./output/statistics"),
            datasets=["eurosat", "imagenet", "midog", "camelyon17wilds", "epistr"],
            against=ag,
            n_bootstrap=10_000,
        )
