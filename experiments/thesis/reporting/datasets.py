"""
Dataset Statistics Extractor for Thesis
----------------------------------------
Iterates across ImageNet, EuroSAT, MIDOG, Camelyon17, EpiStr, and UCMerced splits,
computes sample counts, class distributions, imbalance ratios, Shannon's Equitability,
and image/patch metadata by sampling across multiple items per split.

Run:
    python -m experiments.extract_thesis_stats
    python -m experiments.extract_thesis_stats --out_dir ./thesis_stats
"""

import json
import math
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.data import create_dataset

ROOT_DIR = "./data"

LOGICAL_DATASETS = {
    "ImageNet": {
        "dname": "imagenet",
        "modality": "Natural Images",
        "splits": {"train": "train", "val": "val", "test": None},
    },
    "ImageNet (Subset)": {
        "dname": "imagenet",
        "modality": "Natural Images",
        "splits": {"train": "train@test_r", "val": "val@test_r", "test": None},
    },
    "ImageNet-R": {
        "dname": "imagenet",
        "modality": "Natural Images",
        "splits": {"train": None, "val": None, "test": "test_r"},
    },
    "UCMerced": {
        "dname": "ucmerced",
        "modality": "Remote Sensing",
        "splits": {"train": None, "val": None, "test": "test"},
    },
    "UCMerced (subset)": {
        "dname": "eurosat",
        "modality": "Remote Sensing",
        "splits": {"train": None, "val": None, "test": "ucmerced"},
    },
    "EuroSAT": {
        "dname": "eurosat",
        "modality": "Remote Sensing",
        "splits": {"train": "train", "val": "val", "test": None},
    },
    "EuroSat (subset)": {
        "dname": "eurosat",
        "modality": "Remote Sensing",
        "splits": {"train": "train@ucmerced", "val": "val@ucmerced", "test": None},
    },
    "MIDOG": {
        "dname": "midog",
        "modality": "Histopathology",
        "splits": {"train": "train", "val": "val", "test": "test"},
    },
    "Camelyon17": {
        "dname": "camelyon17wilds",
        "modality": "Histopathology",
        "splits": {"train": "train", "val": "val", "test": "test"},
    },
    "EpiStr": {
        "dname": "epistr",
        "modality": "Histopathology",
        "splits": {"train": "train", "val": "val", "test": "test"},
    },
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


def compute_shannon_equitability(counts: Counter) -> float:
    """
    Computes Shannon's Equitability Index (E_H) from class counts.
    E_H = H / ln(K), where H is Shannon entropy and K is the number of classes.
    """
    total_samples = sum(counts.values())
    if total_samples == 0:
        return 0.0

    probabilities = [c / total_samples for c in counts.values() if c > 0]
    num_classes = len(probabilities)

    if num_classes <= 1:
        return 1.0

    shannon_index = -sum(p * math.log(p) for p in probabilities)
    max_entropy = math.log(num_classes)

    return shannon_index / max_entropy


def compute_split_metrics(ds: Dataset, dataset_name: str, split: str):
    """Computes sample counts, class distributions, imbalance ratios, equitability, and patch metadata."""
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
            "shannon_equitability": 0.0,
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

    equitability = compute_shannon_equitability(counts)

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
        "shannon_equitability": round(equitability, 4),
        **img_meta,
    }


def generate_latex_table(all_results: dict) -> str:
    """Generates LaTeX table grouping datasets by modality with multirow cells."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{l l c c c c}",
        r"\hline",
        r"\textbf{Modality} & \textbf{Dataset} & \textbf{\# Classes} & \textbf{Train / Val / Test} & \textbf{Eq} & \textbf{IR} \\",
        r"\hline",
    ]

    modality_groups = {}
    for logical_name, config in LOGICAL_DATASETS.items():
        mod = config["modality"]
        if mod not in modality_groups:
            modality_groups[mod] = []
        modality_groups[mod].append((logical_name, config))

    total_modalities = len(modality_groups)

    for mod_idx, (modality, items) in enumerate(modality_groups.items()):
        group_size = len(items)

        for i, (logical_name, config) in enumerate(items):
            split_map = config["splits"]

            sample_counts = []
            eq_list = []
            ir_list = []
            max_classes = 0

            for split_type in ["train", "val", "test"]:
                actual_split = split_map[split_type]
                if actual_split and logical_name in all_results and actual_split in all_results[logical_name]:
                    stats = all_results[logical_name][actual_split]
                    sample_counts.append(f"{stats['total_samples']:,}")
                    max_classes = max(max_classes, stats["num_classes"])

                    eq_val = stats["shannon_equitability"]
                    eq_list.append(f"{eq_val:.4f}" if eq_val is not None else "N/A")

                    ir_val = stats["imbalance_ratio"]
                    ir_list.append(f"{ir_val:.2f}" if ir_val is not None else "N/A")
                else:
                    sample_counts.append("-")

            counts_str = " / ".join(sample_counts)
            eq_str = " / ".join(eq_list) if eq_list else "N/A"
            ir_str = " / ".join(ir_list) if ir_list else "N/A"

            modality_cell = f"\\multirow{{{group_size}}}{{*}}{{{modality}}}" if i == 0 else ""

            row = (
                f"{modality_cell} & {logical_name} & {max_classes} & "
                f"{counts_str} & {eq_str} & {ir_str} \\\\"
            )
            lines.append(row)

        if mod_idx < total_modalities - 1:
            lines.append(r"\cline{1-6}")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def run_extraction(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    print("==========================================================================================")
    print("                              EXTRACTING THESIS DATASET STATISTICS                        ")
    print("==========================================================================================")

    for logical_name, config in LOGICAL_DATASETS.items():
        dname = config["dname"]
        all_results[logical_name] = {}

        print(f"\n---> Dataset: {logical_name.upper()}")

        for split_type, actual_split in config["splits"].items():
            if not actual_split:
                continue

            try:
                ds = create_dataset(dname, ROOT_DIR, actual_split)
                stats = compute_split_metrics(ds, logical_name, actual_split)
                all_results[logical_name][actual_split] = stats

                print(
                    f"  Split: {actual_split:<18} | Samples: {stats['total_samples']:<8} | "
                    f"Classes: {stats['num_classes']:<4} | Res: {stats['spatial_resolution']:<15} | "
                    f"Ch: {stats['channels']:<2} | IR: {stats['imbalance_ratio']} | Eq: {stats['shannon_equitability']}"
                )

            except Exception as e:
                print(f"  [ERROR] Failed loading {logical_name} split '{actual_split}': {e}")

    json_path = out_dir / "thesis_dataset_stats.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved JSON Summary]: {json_path}")

    latex_table_str = generate_latex_table(all_results)
    tex_path = out_dir / "dataset_summary_rows.tex"
    with open(tex_path, "w") as f:
        f.write(latex_table_str)
    print(f"[Saved LaTeX Table]: {tex_path}\n")


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
