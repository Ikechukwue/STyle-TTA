"""
Evaluate Predictions from JSON
================================

Reads per-dataset predictions JSON files produced by the TTA inference
pipeline and computes classification metrics (Accuracy, Balanced Accuracy,
AUC, ECE).

Supports two input layouts:

1. **Per-run files** (produced directly by ``run_inference.py``)::

       predictions/<dataset>/<experiment_key>.json
       # Each file contains a single experiment dict with "config",
       # "predictions", "total_samples", "completed".

2. **Merged / legacy files** (produced by ``collect_results.py``)::

       predictions/<dataset>.json
       # A single file containing a dict keyed by experiment_key.

Usage
-----
::

    # Evaluate a single merged per-dataset predictions file
    python -m experiments.evaluate_predictions \\
        --predictions_path results/tta_inference/predictions_merged/epistr.json

    # Evaluate a directory of merged per-dataset files
    python -m experiments.evaluate_predictions \\
        --predictions_dir results/tta_inference/predictions_merged/

    # Evaluate per-run files (directory of directories)
    python -m experiments.evaluate_predictions \\
        --predictions_dir results/tta_inference/predictions/

    # Override output path
    python -m experiments.evaluate_predictions \\
        --predictions_dir results/tta_inference/predictions/ \\
        --output_path results/tta_inference/metrics/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from experiments.data import NUM_CLASSES, TASK_TYPE
from experiments.classifier_evaluation import compute_metrics


# =============================================================================
# Core evaluation
# =============================================================================
def evaluate_experiment(
    exp_key: str,
    data: dict,
    source_file: str,
) -> Optional[Dict]:
    """Compute metrics for a single experiment entry inside a per-dataset file.

    Returns a dict with config + ``metrics``, or ``None`` if incomplete.
    """
    if not data.get("completed", False):
        n_done = len(data.get("predictions", []))
        total = data.get("total_samples", "?")
        print(f"  [SKIP] {exp_key} — not yet completed ({n_done}/{total})")
        return None

    predictions = data.get("predictions", [])
    if not predictions:
        print(f"  [SKIP] {exp_key} — no predictions")
        return None

    config = data.get("config", {})
    dataset = config.get("dataset")
    if dataset is None:
        print(f"  [SKIP] {exp_key} — missing dataset in config")
        return None

    num_classes = NUM_CLASSES.get(dataset)
    task_type = TASK_TYPE.get(dataset)
    if num_classes is None or task_type is None:
        print(f"  [SKIP] {exp_key} — unknown dataset '{dataset}'")
        return None

    y_true = np.array([p["y_true"] for p in predictions])
    y_pred = np.array([p["y_pred"] for p in predictions])

    # Squeeze singleton dimensions
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.squeeze(1)
    if y_pred.ndim == 3 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)

    metrics = compute_metrics(y_true, y_pred, num_classes, task_type)

    return {
        **config,
        "experiment_key": exp_key,
        "source_file": source_file,
        "n_samples": len(predictions),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "metrics": metrics,
    }


def evaluate_prediction_file(pred_path: Path) -> List[Dict]:
    """Evaluate all experiments inside a per-dataset predictions file.

    Handles both formats:
    - **Merged / legacy**: a single JSON with experiment keys at the top level.
    - **Per-run**: the file *is* a single experiment (has "config" at the top level).

    Returns a list of result dicts (one per completed experiment).
    """
    with open(pred_path) as f:
        all_data = json.load(f)

    results: List[Dict] = []

    # Detect per-run format (single experiment) vs. merged format (keyed dict)
    if "config" in all_data and "predictions" in all_data:
        # Per-run file: the entire file is one experiment
        exp_key = pred_path.stem  # filename without .json == experiment_key
        result = evaluate_experiment(exp_key, all_data, str(pred_path))
        if result is not None:
            results.append(result)
    else:
        # Merged / legacy format: keyed dict of experiments
        for exp_key, exp_data in all_data.items():
            result = evaluate_experiment(exp_key, exp_data, str(pred_path))
            if result is not None:
                results.append(result)
    return results


# =============================================================================
# CLI helpers
# =============================================================================
def _collect_prediction_files(
    predictions_path: Optional[str] = None,
    predictions_dir: Optional[str] = None,
) -> List[Path]:
    """Return a list of per-dataset prediction JSON files to evaluate.

    Supports:
    - A single merged JSON file (``--predictions_path``)
    - A directory of merged JSON files (``--predictions_dir``)
    - A directory of per-dataset sub-directories with per-run JSON files
      (``--predictions_dir`` pointing at e.g. ``predictions/``)
    """
    files: List[Path] = []
    if predictions_path:
        p = Path(predictions_path)
        if p.is_file():
            files.append(p)
        else:
            print(f"WARNING: file not found: {p}")
    if predictions_dir:
        d = Path(predictions_dir)
        if d.is_dir():
            # Collect top-level JSON files (merged format)
            files.extend(sorted(d.glob("*.json")))
            # Collect per-run JSON files inside dataset sub-directories
            for sub in sorted(d.iterdir()):
                if sub.is_dir():
                    files.extend(sorted(sub.glob("*.json")))
        else:
            print(f"WARNING: directory not found: {d}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate per-sample predictions from TTA inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions_path", type=str, default=None,
        help="Path to a single per-dataset predictions JSON file",
    )
    parser.add_argument(
        "--predictions_dir", type=str, default=None,
        help="Directory containing per-dataset predictions JSON files",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Directory for output metric JSON files "
             "(defaults to sibling 'metrics/' directory)",
    )
    args = parser.parse_args()

    if args.predictions_path is None and args.predictions_dir is None:
        parser.error(
            "At least one of --predictions_path or "
            "--predictions_dir is required."
        )

    pred_files = _collect_prediction_files(
        args.predictions_path, args.predictions_dir,
    )
    if not pred_files:
        print("No prediction files found.")
        return

    print(f"Found {len(pred_files)} prediction file(s)\n")

    all_results: List[Dict] = []
    for pf in pred_files:
        print(f"Evaluating {pf.name} ...")
        results = evaluate_prediction_file(pf)
        for result in results:
            all_results.append(result)
            m = result["metrics"]
            key = result["experiment_key"]
            print(f"  [{key}]")
            print(f"    Accuracy     : {m['accuracy']:.4f}")
            print(f"    Balanced Acc : {m['balanced_accuracy']:.4f}")
            print(f"    AUC          : {m['auc']:.4f}")
            print(f"    ECE          : {m['ece']:.4f}")
        print()

    if not all_results:
        print("No completed experiments to evaluate.")
        return

    # ---- Determine output directory ----------------------------------------
    if args.output_path:
        out_dir = Path(args.output_path)
    elif args.predictions_dir:
        out_dir = Path(args.predictions_dir).parent / "metrics"
    else:
        out_dir = Path(args.predictions_path).parent.parent / "metrics"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Write per-dataset result files + summary --------------------------
    # Group results by dataset
    by_dataset: Dict[str, Dict] = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        key = r["experiment_key"]
        if ds not in by_dataset:
            by_dataset[ds] = {}
        by_dataset[ds][key] = {
            k: v for k, v in r.items()
            if k not in ("source_file",)
        }

    for ds, experiments in by_dataset.items():
        out_file = out_dir / f"{ds}.json"
        with open(out_file, "w") as f:
            json.dump(experiments, f, indent=2)

    # Summary table
    summary_file = out_dir / "_summary.json"
    summary = []
    for r in all_results:
        summary.append({
            "dataset": r.get("dataset"),
            "classifier": r.get("classifier"),
            "tta_method": r.get("tta_method"),
            "eval_strategy": r.get("eval_strategy"),
            "retrieval_strategy": r.get("retrieval_strategy"),
            "color_method": r.get("color_method"),
            "experiment_key": r.get("experiment_key"),
            "n_samples": r.get("n_samples"),
            **r["metrics"],
        })
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Metrics written to {out_dir}/")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
