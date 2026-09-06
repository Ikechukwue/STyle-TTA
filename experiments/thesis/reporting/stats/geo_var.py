from config.constants import (
    TTA_STRATEGIES,
    ALL_CLASSIFIERS,
    ALL_SEEDS,
    ALL_SPLITS,
    TRUE_SPLITS,
)
from config.helpers import load_json
from pathlib import Path
import numpy as np
import csv

from experiments.data import (
    NUM_CLASSES,
    TASK_TYPE,
)
from experiments.classifier_evaluation import compute_metrics


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


def aggregate_metrics(seed_metrics):
    """
    Aggregate metric dictionaries across seeds.

    Returns mean, sample standard deviation, sample variance,
    minimum, and maximum for every numeric metric.
    """

    if not seed_metrics:
        return {}

    metric_names = seed_metrics[0].keys()

    aggregated = {}

    for metric in metric_names:

        values = np.array(
            [m[metric] for m in seed_metrics],
            dtype=float,
        )

        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "variance": float(np.var(values, ddof=1)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values.tolist(),
        }

    return aggregated


# ============================================================
# CONFIGURATION
# ============================================================

config = TTA_STRATEGIES["geometric_tta"]

all_results = {}

# Output directory
output_dir = Path("results/seed_aggregation")
output_dir.mkdir(parents=True, exist_ok=True)

per_seed_csv = output_dir / "geometric_tta_per_seed.csv"
aggregated_csv = output_dir / "geometric_tta_aggregated.csv"


# ============================================================
# COLLECT RESULTS
# ============================================================

for ds, n_sp in ALL_SPLITS.items():

    split = TRUE_SPLITS[n_sp]

    num_classes = NUM_CLASSES[ds]

    # Dataset-specific class corrections
    if ds == "eurosat":
        if split == "ucmerced":
            num_classes = 4

    if ds == "imagenet":
        if "test_r" in split:
            num_classes = 200

    task_type = TASK_TYPE[ds]

    all_results[ds] = {}

    for cl in ALL_CLASSIFIERS:

        all_results[ds][cl] = {}

        for views in config["axis"]:

            all_results[ds][cl][views] = {}

            seed_metrics = []
            seed_names = []

            y_true_reference = None

            for seed in ALL_SEEDS:

                results = Path(
                    f"results/geometric_tta/tta_inference/results/{ds}/{split}"
                )

                path = config["template"].format(
                    cl=cl,
                    eval="vanilla",
                    rfs=views,
                    seed=seed,
                    retr="",
                )

                results_dir = results / path

                if not results_dir.exists():
                    print(f"Nothing found for: {results_dir}")
                    continue

                loaded_results = load_json(results_dir)

                
                seed_metrics.append(loaded_results["metrics"])
                seed_names.append(seed)

            # Nothing available
            if not seed_metrics:
                continue

            # Aggregate all metrics across seeds
            aggregated = aggregate_metrics(seed_metrics)

            all_results[ds][cl][views] = {
                "seeds": seed_names,
                "per_seed": seed_metrics,
                "aggregated": aggregated,
            }


# ============================================================
# SAVE PER-SEED RESULTS
# ============================================================

per_seed_rows = []

for ds, classifiers in all_results.items():

    for cl, strategies in classifiers.items():

        for views, result in strategies.items():

            for seed, metrics in zip(
                result["seeds"],
                result["per_seed"],
            ):

                row = {
                    "dataset": ds,
                    "classifier": cl,
                    "views": views,
                    "seed": seed,
                }

                # Add every metric
                for metric, value in metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        row[metric] = float(value)

                per_seed_rows.append(row)


if per_seed_rows:

    # Collect all possible columns
    fieldnames = []

    for row in per_seed_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        per_seed_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(per_seed_rows)


# ============================================================
# SAVE AGGREGATED RESULTS
# ============================================================

aggregated_rows = []

for ds, classifiers in all_results.items():

    for cl, strategies in classifiers.items():

        for views, result in strategies.items():

            aggregated = result["aggregated"]

            # Create one row for each dataset/classifier/view
            row = {
                "dataset": ds,
                "classifier": cl,
                "views": views,
                "n_seeds": len(result["seeds"]),
            }

            # Add mean/std/variance/min/max
            for metric, stats in aggregated.items():

                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
                row[f"{metric}_variance"] = stats["variance"]
                row[f"{metric}_min"] = stats["min"]
                row[f"{metric}_max"] = stats["max"]

            aggregated_rows.append(row)


if aggregated_rows:

    fieldnames = []

    for row in aggregated_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        aggregated_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(aggregated_rows)


# ============================================================
# PRINT RESULTS
# ============================================================

for ds, classifiers in all_results.items():

    print("\n")
    print("=" * 100)
    print(f"DATASET: {ds}")
    print("=" * 100)

    for cl, strategies in classifiers.items():

        print("\n")
        print(f"CLASSIFIER: {cl}")
        print("-" * 100)

        for views, result in strategies.items():

            print(f"\nGeometric TTA: {views}")
            print(f"Seeds: {result['seeds']}")
            print()

            aggregated = result["aggregated"]
            per_seed = result["per_seed"]

            # Per-seed results
            print("Per-seed metrics:")

            for seed, metrics in zip(
                result["seeds"],
                per_seed,
            ):

                print(f"\n  Seed {seed}:")

                for metric, value in metrics.items():

                    if isinstance(value, (int, float, np.number)):

                        print(
                            f"    {metric:<30} "
                            f"{float(value):.6f}"
                        )

            # Aggregate results
            print("\nAggregate across seeds:")

            print(
                f"  {'Metric':<30}"
                f"{'Mean':>12}"
                f"{'Std':>12}"
                f"{'Variance':>15}"
                f"{'Min':>12}"
                f"{'Max':>12}"
            )

            print("  " + "-" * 91)

            for metric, stats in aggregated.items():

                print(
                    f"  {metric:<30}"
                    f"{stats['mean']:>12.6f}"
                    f"{stats['std']:>12.6f}"
                    f"{stats['variance']:>15.6f}"
                    f"{stats['min']:>12.6f}"
                    f"{stats['max']:>12.6f}"
                )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 100)
print("RESULTS SAVED")
print("=" * 100)
print(f"Per-seed results:  {per_seed_csv}")
print(f"Aggregated results: {aggregated_csv}")
