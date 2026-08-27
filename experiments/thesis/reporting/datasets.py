"""
Dataset Statistics Extractor for Thesis
----------------------------------------
Iterates across ImageNet, EuroSAT, MIDOG, Camelyon17, and EPIStress splits,
computes sample counts, class distributions, imbalance ratios, and image/patch metadata
by sampling across multiple items per split.

Run:
    python -m experiments.extract_thesis_stats
    python -m experiments.extract_thesis_stats --out_dir ./thesis_stats
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.data import create_dataset

ROOT_DIR = "./data"

DATASET_CONFIGS = {
    "imagenet": [
        "train",
        "val",
        "train@test_r",
        "val@test_r",
        "test_r",
    ],
    "eurosat": [
        "train",
        "val",
        "test",
        "ucmerced",
        "train@ucmerced",
        "val@ucmerced",
        "test@ucmerced",
    ],
    "midog": [
        "train",
        "val",
        "test",
    ],
    "camelyon17wilds": [
        "train",
        "val",
        "test",
    ],
    "epistr": [
        "train",
        "val",
        "test",
    ],
}


def unwrap_dataset(ds: Dataset) -> Dataset:
    """Recursively unwraps dataset wrappers."""
    current = ds
    while hasattr(current, "dataset"):
        current = current.dataset
    return current


def extract_samples_and_labels(ds: Dataset):
    """Extracts labels without iterating through full image loads where possible."""
    base_ds = unwrap_dataset(ds)

    if hasattr(base_ds, "samples") and base_ds.samples:
        return [lbl for _, lbl in base_ds.samples]

    if hasattr(base_ds, "targets") and base_ds.targets is not None:
        targets = base_ds.targets
        return targets.tolist() if hasattr(targets, "tolist") else list(targets)

    if hasattr(base_ds, "y_array") and base_ds.y_array is not None:
        y = base_ds.y_array
        return y.tolist() if hasattr(y, "tolist") else list(y)

    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        parent_labels = extract_samples_and_labels(ds.dataset)
        if parent_labels:
            return [parent_labels[i] for i in ds.indices]

    return None


def extract_image_metadata(ds: Dataset, num_checks: int = 50):
    """
    Samples multiple items across the dataset to determine image/patch dimensions,
    channels, tensor shapes, and detect if resolutions vary across samples.
    """
    total_samples = len(ds)
    if total_samples == 0:
        return {
            "tensor_shape": "N/A",
            "channels": "N/A",
            "spatial_resolution": "N/A",
            "native_resolution": "N/A",
            "is_fixed_resolution": True,
        }

    base_ds = unwrap_dataset(ds)
    native_res = getattr(base_ds, "native_resolution", None)

    indices = np.linspace(0, total_samples - 1, num=min(num_checks, total_samples), dtype=int)

    shapes = []
    resolutions = []
    channels_set = set()

    for idx in indices:
        try:
            sample, _ = ds[idx]
        except Exception:
            continue

        if isinstance(sample, torch.Tensor) or hasattr(sample, "shape"):
            shape = tuple(sample.shape)
            shapes.append(shape)
            if len(shape) == 3:
                channels_set.add(shape[0])
                resolutions.append((shape[1], shape[2]))
            elif len(shape) == 2:
                channels_set.add(1)
                resolutions.append((shape[0], shape[1]))
        elif hasattr(sample, "size"):
            w, h = sample.size
            resolutions.append((h, w))
            mode_map = {"RGB": 3, "L": 1, "RGBA": 4}
            ch = mode_map.get(getattr(sample, "mode", ""), "N/A")
            if ch != "N/A":
                channels_set.add(ch)

    if not resolutions:
        return {
            "tensor_shape": "N/A",
            "channels": "N/A",
            "spatial_resolution": "N/A",
            "native_resolution": str(native_res) if native_res else "N/A",
            "is_fixed_resolution": True,
        }

    unique_res = set(resolutions)
    is_fixed = len(unique_res) == 1

    if is_fixed:
        h, w = resolutions[0]
        spatial_size = f"{w}x{h}"
    else:
        min_h = min(r[0] for r in resolutions)
        max_h = max(r[0] for r in resolutions)
        min_w = min(r[1] for r in resolutions)
        max_w = max(r[1] for r in resolutions)
        spatial_size = f"{min_w}x{min_h} to {max_w}x{max_h}"

    ch_str = "/".join(str(c) for c in sorted(channels_set)) if channels_set else "N/A"
    tensor_shape_str = str(shapes[0]) if is_fixed and shapes else "Variable"

    if native_res is None:
        native_res = spatial_size

    return {
        "tensor_shape": tensor_shape_str,
        "channels": ch_str,
        "spatial_resolution": spatial_size,
        "native_resolution": str(native_res),
        "is_fixed_resolution": is_fixed,
    }


def compute_split_metrics(ds: Dataset, dataset_name: str, split: str):
    """Computes sample counts, class distributions, imbalance ratios, and patch metadata."""
    total_samples = len(ds)
    img_meta = extract_image_metadata(ds)

    if total_samples == 0:
        return {
            "dataset": dataset_name,
            "split": split,
            "total_samples": 0,
            "num_classes": 0,
            "class_counts": {},
            "imbalance_ratio": None,
            **img_meta,
        }

    labels = extract_samples_and_labels(ds)

    if labels is None:
        print(f"  [info] Iterating directly through {split} to collect targets...")
        labels = []
        for i in range(total_samples):
            _, target = ds[i]
            labels.append(int(target))

    counts = Counter(labels)
    num_classes = len(counts)

    min_c = min(counts.values()) if counts else 0
    max_c = max(counts.values()) if counts else 0
    imbalance_ratio = (max_c / min_c) if min_c > 0 else float("inf")

    return {
        "dataset": dataset_name,
        "split": split,
        "total_samples": total_samples,
        "num_classes": num_classes,
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "min_per_class": min_c,
        "max_per_class": max_c,
        "mean_per_class": float(np.mean(list(counts.values()))) if counts else 0.0,
        "imbalance_ratio": round(imbalance_ratio, 2) if min_c > 0 else None,
        **img_meta,
    }


def format_latex_row(stats: dict) -> str:
    """Formats a dictionary of stats into a LaTeX table row."""
    d_name = stats["dataset"]
    split = stats["split"].replace("_", r"\_")
    n_samples = f"{stats['total_samples']:,}"
    n_classes = stats["num_classes"]
    ir = stats["imbalance_ratio"]
    ir_str = f"{ir:.2f}" if ir is not None else "N/A"
    res = stats["spatial_resolution"]
    channels = stats["channels"]

    return (
        f"{d_name} & \\texttt{{{split}}} & {n_classes} & {n_samples} & "
        f"{res} & {channels} & {ir_str} \\\\"
    )


def run_extraction(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    latex_rows = []

    print("==========================================================================================")
    print("                             EXTRACTING THESIS DATASET STATISTICS                         ")
    print("==========================================================================================")

    for dname, splits in DATASET_CONFIGS.items():
        all_results[dname] = {}
        print(f"\n---> Dataset: {dname.upper()}")

        for split in splits:
            try:
                ds = create_dataset(dname, ROOT_DIR, split)
                stats = compute_split_metrics(ds, dname, split)
                all_results[dname][split] = stats

                row = format_latex_row(stats)
                latex_rows.append(row)

                print(
                    f"  Split: {split:<18} | Samples: {stats['total_samples']:<8} | "
                    f"Classes: {stats['num_classes']:<4} | Res: {stats['spatial_resolution']:<15} | "
                    f"Ch: {stats['channels']:<2} | IR: {stats['imbalance_ratio']}"
                )

            except Exception as e:
                print(f"  [ERROR] Failed loading {dname} split '{split}': {e}")

    json_path = out_dir / "thesis_dataset_stats.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved JSON Summary]: {json_path}")

    tex_path = out_dir / "dataset_summary_rows.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_rows))
    print(f"[Saved LaTeX Table Rows]: {tex_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dataset statistics for thesis.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./results/thesis_stats_output",
        help="Directory to store JSON and LaTeX outputs.",
    )
    args = parser.parse_args()

    run_extraction(Path(args.out_dir))
