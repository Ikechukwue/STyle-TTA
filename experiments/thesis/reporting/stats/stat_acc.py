from pathlib import Path
from typing import Optional
import json
from tqdm import tqdm 
import numpy as np
from scipy.stats import chi2


def get_predictions(results: dict):
    """
    Extract true labels and predicted classes from a results JSON.

    Assumes:
        results["predictions"] = [
            {"y_true": ..., "y_pred": [...]},
            ...
        ]

    Returns:
        y_true: shape (N,)
        y_pred: shape (N,)
    """
    y_true = np.array(
        [sample["y_true"] for sample in results["predictions"]]
    )

    y_pred_matrix = np.array(
        [sample["y_pred"] for sample in results["predictions"]]
    )

    y_pred = np.argmax(y_pred_matrix, axis=1)

    return y_true, y_pred


def balanced_accuracy_score_np(y_true, y_pred):
    """
    Balanced accuracy = mean recall over classes.
    """
    classes = np.unique(y_true)

    recalls = []

    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) == 0:
            continue

        recall = np.mean(y_pred[mask] == cls)
        recalls.append(recall)

    return float(np.mean(recalls))


def stratified_bootstrap_difference(
    y_true,
    pred_baseline,
    pred_tta,
    n_bootstrap=10_000,
    confidence_level=0.95,
    seed=42,
):
    """
    Stratified bootstrap CI for the difference in balanced accuracy:

        TTA balanced accuracy - baseline balanced accuracy

    Resampling is performed independently within each ground-truth class.
    """

    rng = np.random.default_rng(seed)

    classes = np.unique(y_true)

    # Store indices for every class.
    class_indices = {
        cls: np.where(y_true == cls)[0]
        for cls in classes
    }

    bootstrap_differences = np.empty(n_bootstrap)

    for b in tqdm(range(n_bootstrap)):

        sampled_indices = []

        # Stratified resampling:
        # sample the same number of examples from each class
        # as originally present in that class.
        for cls in classes:
            indices = class_indices[cls]

            sampled = rng.choice(
                indices,
                size=len(indices),
                replace=True,
            )

            sampled_indices.append(sampled)

        sampled_indices = np.concatenate(sampled_indices)

        y_true_b = y_true[sampled_indices]
        baseline_b = pred_baseline[sampled_indices]
        tta_b = pred_tta[sampled_indices]

        baseline_bal_acc = balanced_accuracy_score_np(
            y_true_b,
            baseline_b,
        )

        tta_bal_acc = balanced_accuracy_score_np(
            y_true_b,
            tta_b,
        )

        bootstrap_differences[b] = (
            tta_bal_acc - baseline_bal_acc
        )

    alpha = 1.0 - confidence_level

    lower = np.percentile(
        bootstrap_differences,
        100 * (alpha / 2),
    )

    upper = np.percentile(
        bootstrap_differences,
        100 * (1 - alpha / 2),
    )

    # Difference on the original test set
    baseline_bal_acc = balanced_accuracy_score_np(
        y_true,
        pred_baseline,
    )

    tta_bal_acc = balanced_accuracy_score_np(
        y_true,
        pred_tta,
    )

    observed_difference = tta_bal_acc - baseline_bal_acc

    return {
        "baseline_balanced_accuracy": baseline_bal_acc,
        "tta_balanced_accuracy": tta_bal_acc,
        "difference": observed_difference,
        "difference_percentage_points": observed_difference * 100,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_lower_percentage_points": float(lower * 100),
        "ci_upper_percentage_points": float(upper * 100),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def mcnemar_test(
    y_true,
    pred_baseline,
    pred_tta,
):
    """
    McNemar's test comparing paired baseline and TTA predictions.

    Uses the exact binomial test, which is preferable to the
    chi-square approximation when the number of discordant
    observations is small.
    """

    baseline_correct = pred_baseline == y_true
    tta_correct = pred_tta == y_true

    # Baseline correct, TTA wrong
    b = np.sum(
        baseline_correct & ~tta_correct
    )

    # Baseline wrong, TTA correct
    c = np.sum(
        ~baseline_correct & tta_correct
    )

    discordant = b + c

    if discordant == 0:
        p_value = 1.0
    else:
        # Exact two-sided binomial test:
        # Under H0, b and c have probability 0.5 each.
        from scipy.stats import binomtest

        p_value = binomtest(
            k=min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    return {
        "baseline_correct_tta_wrong": int(b),
        "baseline_wrong_tta_correct": int(c),
        "discordant_pairs": int(discordant),
        "p_value": float(p_value),
    }


def calculate_statistical_comparison(
    baseline_path: Path,
    tta_path: Path,
    output_path: Path,
    n_bootstrap=10_000,
    confidence_level=0.95,
    seed=42,
):
    """
    Calculate stratified bootstrap CI and McNemar's test
    for one baseline-vs-TTA comparison.

    Results are saved as JSON.
    """

    with open(baseline_path, "r") as f:
        baseline_results = json.load(f)

    with open(tta_path, "r") as f:
        tta_results = json.load(f)

    y_true_baseline, pred_baseline = get_predictions(
        baseline_results
    )

    y_true_tta, pred_tta = get_predictions(
        tta_results
    )


    if len(y_true_baseline) != len(y_true_tta):
        raise ValueError(
            "Baseline and TTA contain different numbers of predictions."
        )

    if not np.array_equal(y_true_baseline, y_true_tta):
        raise ValueError(
            "Ground-truth labels do not match between baseline and TTA."
        )

    y_true = y_true_baseline

    bootstrap_results = stratified_bootstrap_difference(
        y_true=y_true,
        pred_baseline=pred_baseline,
        pred_tta=pred_tta,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )

    mcnemar_results = mcnemar_test(
        y_true=y_true,
        pred_baseline=pred_baseline,
        pred_tta=pred_tta,
    )

    result = {
        "baseline_file": str(baseline_path),
        "tta_file": str(tta_path),
        "n_samples": int(len(y_true)),
        "n_classes": int(len(np.unique(y_true))),

        "bootstrap": bootstrap_results,

        "mcnemar": mcnemar_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(f"Saved statistical results to: {output_path}")

    return result

if __name__ == "__main__":
    baseline_path = "/home/stud/nemmler/retristyle/results/geometric_tta/tta_inference/predictions/imagenet/test_r/densenet121_geometric_vanilla_nviews1_seed71397589.json"
    tta_path= "/home/stud/nemmler/retristyle/results/geometric_tta/tta_inference/predictions/imagenet/test_r/densenet121_geometric_vanilla_nviews64_seed71397589.json"
    calculate_statistical_comparison(
    baseline_path=Path(baseline_path),
    tta_path=Path(tta_path),
    output_path=Path("./results/statistics/styleid_vs_baseline.json"),
)
