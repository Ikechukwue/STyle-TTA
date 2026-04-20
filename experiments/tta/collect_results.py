"""
Collect per-run JSON files into combined per-dataset files.
============================================================

After running TTA inference on a cluster (where each job writes its own
file to avoid contention), use this script to merge everything back into
a single per-dataset predictions and results file.

**Input layout** (produced by :mod:`experiments.tta.run_inference`)::

    <results>/tta_inference/predictions/<dataset>/<experiment_key>.json
    <results>/tta_inference/results/<dataset>/<experiment_key>.json

**Output layout** (one merged file per dataset)::

    <results>/tta_inference/predictions_merged/<dataset>.json
    <results>/tta_inference/results_merged/<dataset>.json

The merged predictions file has the legacy keyed structure::

    { "<experiment_key>": { "config": {...}, "predictions": [...], ... }, ... }

The merged results file::

    { "<experiment_key>": { "dataset": "...", "metrics": {...}, ... }, ... }

Usage
-----
::

    # Merge all datasets found under the default results dir
    python -m experiments.tta.collect_results --results_dir ./results

    # Merge a specific dataset only
    python -m experiments.tta.collect_results --results_dir ./results --dataset epistr

    # Custom output directory
    python -m experiments.tta.collect_results --results_dir ./results \\
        --output_dir ./results/tta_inference/merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data: dict, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def collect_predictions(
    pred_dir: Path, output_path: Path, *, verbose: bool = True,
) -> int:
    """Merge per-run prediction files from *pred_dir* into one keyed file.

    Returns the number of experiments merged.
    """
    merged: dict = {}
    files = sorted(pred_dir.glob("*.json"))
    for f in files:
        key = f.stem  # filename without .json  == experiment_key
        data = _load_json(f)
        merged[key] = data

    if merged:
        _save_json(output_path, merged, indent=None)  # compact (large file)
        if verbose:
            print(f"  ✓ {output_path.name}: {len(merged)} experiment(s)")
    return len(merged)


def collect_results(
    res_dir: Path, output_path: Path, *, verbose: bool = True,
) -> int:
    """Merge per-run result files from *res_dir* into one keyed file.

    Returns the number of experiments merged.
    """
    merged: dict = {}
    files = sorted(res_dir.glob("*.json"))
    for f in files:
        key = f.stem
        data = _load_json(f)
        merged[key] = data

    if merged:
        _save_json(output_path, merged, indent=2)
        if verbose:
            print(f"  ✓ {output_path.name}: {len(merged)} experiment(s)")
    return len(merged)


def collect_all(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    dataset: str | None = None,
    verbose: bool = True,
) -> None:
    """Merge individual per-run files back into combined per-dataset files.

    Parameters
    ----------
    results_dir:
        Top-level results directory (contains ``tta_inference/``).
    output_dir:
        Where to write the merged files.  Defaults to
        ``<results_dir>/tta_inference/predictions_merged/`` and
        ``<results_dir>/tta_inference/results_merged/``.
    dataset:
        If given, only merge this dataset.  Otherwise merge all found.
    """
    base = Path(results_dir) / "tta_inference"
    pred_base = base / "predictions"
    res_base = base / "results"

    if output_dir is not None:
        out = Path(output_dir)
        pred_out_base = out / "predictions"
        res_out_base = out / "results"
    else:
        pred_out_base = base / "predictions_merged"
        res_out_base = base / "results_merged"

    # Discover datasets (sub-directories)
    if dataset:
        datasets = [dataset]
    else:
        datasets = sorted(
            {d.name for d in list(pred_base.iterdir()) + list(res_base.iterdir())
             if d.is_dir()}
        ) if pred_base.exists() or res_base.exists() else []

    if not datasets:
        print("No per-run files found to merge.")
        return

    total_preds = 0
    total_results = 0

    for ds in datasets:
        if verbose:
            print(f"\nDataset: {ds}")

        # Predictions
        pred_dir = pred_base / ds
        if pred_dir.is_dir() and any(pred_dir.glob("*.json")):
            pred_out = pred_out_base / f"{ds}.json"
            total_preds += collect_predictions(pred_dir, pred_out, verbose=verbose)
        elif verbose:
            print(f"  (no prediction files for {ds})")

        # Results
        res_dir = res_base / ds
        if res_dir.is_dir() and any(res_dir.glob("*.json")):
            res_out = res_out_base / f"{ds}.json"
            total_results += collect_results(res_dir, res_out, verbose=verbose)
        elif verbose:
            print(f"  (no result files for {ds})")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Merged predictions : {total_preds} experiments")
        print(f"Merged results     : {total_results} experiments")
        print(f"Predictions dir    : {pred_out_base}")
        print(f"Results dir        : {res_out_base}")
        print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-run prediction/result JSON files into "
                    "combined per-dataset files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Top-level results directory (contains tta_inference/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for merged files "
             "(defaults to <results_dir>/tta_inference/{predictions,results}_merged/)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Merge only this dataset (default: all datasets found)",
    )
    args = parser.parse_args()

    collect_all(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
