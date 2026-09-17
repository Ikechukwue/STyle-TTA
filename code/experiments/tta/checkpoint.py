"""
Checkpoint / resume helpers for per-sample predictions.

Each experiment run writes to its **own** pair of files so that
parallel SLURM jobs never contend on the same file::

    <output>/tta_inference/predictions/<dataset>/<experiment_key>.json
    <output>/tta_inference/results/<dataset>/<experiment_key>.json

Use :mod:`experiments.tta.collect_results` to merge the individual files
into combined per-dataset JSON files after all runs have completed.

Predictions file structure (one per run)::

    {
      "config": { ... },
      "predictions": [
        {"sample_idx": 0, "y_true": 3, "y_pred": [0.1, 0.9]},
        ...
      ],
      "total_samples": 1376,
      "completed": true,
      "elapsed_seconds": 123.4
    }

Results file structure (one per run)::

    {
      "dataset": "epistr",
      "classifier": "densenet121",
      ...
      "metrics": { "accuracy": 0.65, ... }
    }

Files are written atomically (write to ``.tmp``, then ``rename``) so that
a crash or SLURM timeout never leaves a corrupted checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ======================================================================
# Experiment key / tag helpers
# ======================================================================
def build_experiment_key(args: argparse.Namespace, eval_split: str, model:str) -> str:
    """Return a dataset-independent key that uniquely identifies the config.

    Used as the top-level key inside the per-dataset JSON files.

    Key format examples::

        densenet121_retristyle_zero_dino_nrefs7_none_seed71397589
        densenet121_geometric_zero_nviews64_none_seed71397589
        densenet121_geometric_zero_nviews8_color_jitter_seed71397589
        densenet121_tent_vanilla_none_seed71397589
    """
    key = f"{model}_{args.tta_method}_{args.eval_strategy}"
    if args.tta_method in ("adain_tta", "color_tta", "retristyle"):
        key += f"_{args.retrieval_strategy}"
    if args.tta_method == "color_tta":
        key += f"_{args.color_method}"
    # Include view/ref count so sweeps over n_refs / n_views get
    # separate entries in the JSON files.
    if args.tta_method in ("adain_tta", "color_tta", "retristyle"):
        key += f"_nrefs{args.n_refs}"
    elif hasattr(args, "n_views") and args.tta_method not in ("tent", "none"):
        key += f"_nviews{args.n_views}"
    # Include the training augmentation so that classifiers trained with
    # different augmentations produce distinct keys.
    #train_aug = getattr(args, "train_aug", "none")
    #key += f"_{train_aug}"
    key += f"_seed{args.seed}"
    return key


def build_experiment_tag(args: argparse.Namespace, eval_split: str) -> str:
    """Full tag **including** the dataset name (useful for logging)."""
    return f"{args.dataset}_{build_experiment_key(args, eval_split)}"


# ======================================================================
# Path helpers
# ======================================================================
def predictions_path(
    args: argparse.Namespace, eval_split: str, key: str | None = None,
) -> Path:
    """Per-run predictions file (or per-dataset directory when *key* is ``None``).

    With *key*::

        <output>/tta_inference/predictions/<dataset>/<key>.json

    Without *key* (useful for globbing / merging)::

        <output>/tta_inference/predictions/<dataset>/
    """
    base = Path(args.output_path) / "tta_inference" / "predictions" / args.dataset
    if args.split is not None:
        base = base / args.split
    if key is not None:
        return base / f"{key}.json"
    return base


def results_path(
    args: argparse.Namespace, eval_split: str, key: str | None = None,
) -> Path:
    """Per-run results file (or per-dataset directory when *key* is ``None``).

    With *key*::

        <output>/tta_inference/results/<dataset>/<key>.json

    Without *key* (useful for globbing / merging)::

        <output>/tta_inference/results/<dataset>/
    """
    base = Path(args.output_path) / "tta_inference" / "results" / args.dataset
    if args.split is not None:
        base = base / f"{args.split}"
    if key is not None:
        return base / f"{key}.json"
    return base


# ======================================================================
# Atomic JSON helpers (read-modify-write the whole file)
# ======================================================================
def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json_atomic(path: Path, data: dict, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent)
    tmp.rename(path)


# ======================================================================
# Predictions  (checkpoint / resume)
# ======================================================================
def load_predictions(path: Path) -> dict:
    """Load one experiment's predictions from its per-run file.

    Returns an empty skeleton when the file does not exist yet.
    """
    if path.exists():
        return _load_json(path)
    print("No previous prediction attempt found")
    return {"config": {}, "predictions": [], "total_samples": None, "completed": False}


def save_predictions(path: Path, data: dict) -> None:
    """Atomically write one experiment's predictions to its per-run file."""
    _save_json_atomic(path, data)


# ======================================================================
# Results  (metrics summary)
# ======================================================================
def load_results(path: Path) -> dict:
    """Load a single result from a per-run results file."""
    return _load_json(path)


def save_result(path: Path, result: dict) -> None:
    """Atomically write one experiment's result to its per-run file."""
    _save_json_atomic(path, result, indent=2)
