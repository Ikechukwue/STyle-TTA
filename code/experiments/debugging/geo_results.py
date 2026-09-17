from code.config.constants import (
    TTA_STRATEGIES,
    ALL_CLASSIFIERS,
    ALL_SEEDS,
    ALL_SPLITS,
    TRUE_SPLITS,
)
from code.config.helpers import load_json
from pathlib import Path
import json
import shutil
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

config = TTA_STRATEGIES["geometric_tta"]

RESULTS_ROOT = Path(
    "results/geometric_tta/tta_inference/results"
)

# Set to True if you want to keep a backup of the original file
CREATE_BACKUP = True


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_metrics(seed_metrics):
    """
    Aggregate metric dictionaries across seeds.

    Returns a dictionary containing:
        mean
        std
        variance
        min
        max
        values

    for every metric.
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
# PROCESS ALL DATASETS / CLASSIFIERS / VIEW COUNTS
# ============================================================

for ds, n_sp in ALL_SPLITS.items():

    split = TRUE_SPLITS[n_sp]

    print("\n")
    print("=" * 100)
    print(f"DATASET: {ds}")
    print("=" * 100)

    for cl in ALL_CLASSIFIERS:

        print(f"\nCLASSIFIER: {cl}")
        print("-" * 100)

        for views in config["axis"]:

            seed_results = []
            seed_paths = []
            seed_names = []

            # ------------------------------------------------
            # Load all seeds
            # ------------------------------------------------

            for seed in ALL_SEEDS:

                results_dir = Path(
                    f"results/geometric_tta/"
                    f"tta_inference/results/"
                    f"{ds}/{split}"
                )

                path = config["template"].format(
                    cl=cl,
                    eval="vanilla",
                    rfs=views,
                    seed=seed,
                    retr="",
                )

                results_path = results_dir / path

                if not results_path.exists():

                    print(
                        f"  Missing seed {seed}: "
                        f"{results_path}"
                    )

                    continue

                results = load_json(results_path)

                seed_results.append(results)
                seed_paths.append(results_path)
                seed_names.append(seed)

            # ------------------------------------------------
            # Check that we have results
            # ------------------------------------------------

            if not seed_results:
                print(f"  No results found for {views} views.")
                continue

            if len(seed_results) < 2:
                print(
                    f"  Only {len(seed_results)} seed available "
                    f"for {views} views. Skipping aggregation."
                )
                continue

            # ------------------------------------------------
            # Extract metrics
            # ------------------------------------------------

            seed_metrics = [
                result["metrics"]
                for result in seed_results
            ]

            # ------------------------------------------------
            # Aggregate metrics
            # ------------------------------------------------

            aggregated_metrics = aggregate_metrics(
                seed_metrics
            )

            # ------------------------------------------------
            # Print summary
            # ------------------------------------------------

            print(
                f"\n  Geometric TTA: {views} views"
            )
            print(
                f"  Seeds: {seed_names}"
            )

            for metric, stats in aggregated_metrics.items():

                print(
                    f"    {metric:<30}"
                    f"mean={stats['mean']:.6f}  "
                    f"std={stats['std']:.6f}"
                )

            # ------------------------------------------------
            # Update EACH seed file
            #
            # The first/main seed gets:
            #
            #     metrics = mean across seeds
            #
            # and:
            #
            #     aggregated_metrics = full statistics
            #
            # Other fields remain untouched.
            # ------------------------------------------------

            mean_metrics = {
                metric: stats["mean"]
                for metric, stats
                in aggregated_metrics.items()
            }

            for results, results_path in zip(
                seed_results,
                seed_paths,
            ):

                # ------------------------------------------------
                # Optional backup
                # ------------------------------------------------

                if CREATE_BACKUP:

                    backup_path = Path(
                        str(results_path) + ".backup"
                    )

                    if not backup_path.exists():

                        shutil.copy2(
                            results_path,
                            backup_path,
                        )

                # ------------------------------------------------
                # Replace metrics with mean metrics
                # ------------------------------------------------

                results["metrics"] = mean_metrics

                # ------------------------------------------------
                # Add full aggregation information
                # ------------------------------------------------

                results["aggregated_metrics"] = (
                    aggregated_metrics
                )

                # ------------------------------------------------
                # Save
                # ------------------------------------------------

                with open(
                    results_path,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        results,
                        f,
                        indent=4,
                    )

            print(
                f"  Updated {len(seed_paths)} result files."
            )


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)
