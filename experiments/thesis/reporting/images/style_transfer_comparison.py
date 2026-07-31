import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from config.helpers import inject_stylized_images_inplace
from experiments.data import create_dataset
from experiments.metrics.color_metrics import (
    batch_color_moment_distance,
    batch_histogram_distances,
    compute_color_moment_distance,
)
from experiments.metrics.content_metrics import (
    batch_sobel_edge_similarity,
    batch_ssim,
)
from experiments.metrics.model import pixel_model_analysis
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio


def _save_json(output_path: Path, results):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def calculate_batched_metrics(images_A: List[torch.Tensor], images_B: List[torch.Tensor], device="cuda") -> List[Dict]:
    """
    Computes 1:1 metrics for matched pairs between set A and stylized set B.
    """
    if not images_A or not images_B:
        return []

    batch_A = torch.stack(images_A).to(device)
    batch_B = torch.stack(images_B).to(device)

    with torch.no_grad():
        ssim_scores = batch_ssim(batch_A, batch_B, one_to_one=True).cpu().numpy()
        edge_scores = batch_sobel_edge_similarity(batch_A, batch_B, one_to_one=True).cpu().numpy()
        color_scores = batch_color_moment_distance(batch_A, batch_B, one_to_one=True).cpu().numpy()
        kl_dist, js_dist, chi_dist, hist_int = batch_histogram_distances(batch_A, batch_B, one_to_one=True)
        hist_scores = hist_int.cpu().numpy()

    torch.cuda.empty_cache()

    batched_pair_results = []
    N = len(images_A)

    for i in range(N):
        batched_pair_results.append({
            "ssim": float(ssim_scores[i]),
            "edge_similarity": float(edge_scores[i]),
            "color_moment_distance": float(color_scores[i]),
            "histogramm_distance": float(hist_scores[i]),
        })

    return batched_pair_results


def group_by_class(dataset, max_samples, max_cls):
    groups = defaultdict(list)
    labels = getattr(dataset, 'targets', None)
    if labels is None:
        labels = dataset.dataset.dataset.targets

    for i, label in enumerate(labels):
        if not max_cls or (int(label) in groups or len(groups.keys()) < max_cls):
            if not max_samples or len(groups[int(label)]) < max_samples:
                groups[int(label)].append(i)

    return groups


def prepare_datasets(args):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=256),
    ])

    set_A = create_dataset(
        args.dataset,
        args.data_path,
        args.split,
        transform=transform,
    )

    set_B = create_dataset(
        args.dataset,
        args.data_path,
        args.split,
        transform=transform,
    )

    inject_stylized_images_inplace(
        set_B,
        new_base_dir_path=args.stylized_dir,
        view_name=args.view_name
    )

    return set_A, set_B


def aggregate_class_metrics(class_metrics: list) -> dict:
    if not class_metrics:
        return {"n_pairs": 0}

    metric_keys = set()
    for m in class_metrics:
        metric_keys.update(m.keys())

    summary = {}
    for key in metric_keys:
        valid_values = [m[key] for m in class_metrics if m.get(key) is not None]
        if valid_values:
            summary[f"{key}_mean"] = float(np.mean(valid_values))
            summary[f"{key}_std"] = float(np.std(valid_values))
            summary[f"{key}_min"] = float(np.min(valid_values))
            summary[f"{key}_max"] = float(np.max(valid_values))
        else:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            summary[f"{key}_min"] = None
            summary[f"{key}_max"] = None

    summary["n_pairs"] = len(class_metrics)
    return summary


def classes_analysis(set_A, set_B, args):
    max_samples = 0
    max_cls = args.max_classes or 0

    groups_A = group_by_class(set_A, max_samples, max_cls)
    groups_B = group_by_class(set_B, max_samples, max_cls)

    common_classes = set(groups_A.keys()).intersection(set(groups_B.keys()))

    balanced_groups_A = {}
    balanced_groups_B = {}
    for cls in common_classes:
        min_samples = min(len(groups_A[cls]), len(groups_B[cls]))
        if min_samples > 0:
            balanced_groups_A[cls] = groups_A[cls][:min_samples]
            balanced_groups_B[cls] = groups_B[cls][:min_samples]

    groups_A = balanced_groups_A
    groups_B = balanced_groups_B

    all_metrics = []
    class_summaries = {}
    total_pairs = sum(len(v) for v in groups_A.values())

    print(f"Processing {len(common_classes)} classes with total {total_pairs} 1:1 image pairs.")

    models = ["lpips", "ldc", "depthanything_v2_large", "dpt_large"]

    model_metrics = pixel_model_analysis(
        set_A, set_B, common_classes, groups_A, groups_B, models, intra=False, one_to_one=True
    )

    for cls in tqdm(common_classes, desc="Processing Classes"):
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_metrics = calculate_batched_metrics(images_A, images_B, device=device)

        if class_metrics and cls in model_metrics:
            for metric_name, flat_scores in model_metrics[cls].items():
                for idx, pair_dict in enumerate(class_metrics):
                    if idx < len(flat_scores):
                        val = flat_scores[idx]
                        pair_dict[metric_name] = float(val) if val is not None else None

        all_metrics.extend(class_metrics)
        class_summaries[str(cls)] = aggregate_class_metrics(class_metrics)

    summary = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            vals = [m[key] for m in all_metrics if m.get(key) is not None]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_std"] = float(np.std(vals))

    return {"global_summary": summary, "class_summaries": class_summaries}


def domain_analysis(args):
    output_dir = Path(args.output_dir)
    result_dir = output_dir / f"{args.dataset}_{args.split}"
    result_dir.mkdir(parents=True, exist_ok=True)

    set_A, set_B = prepare_datasets(args)
    class_results = classes_analysis(set_A, set_B, args)

    result_name = f"paired_{args.max_classes}_{args.max_samples}_{args.dataset}_{args.split}.json"
    _save_json(result_dir / result_name, class_results)


def get_args():
    parser = argparse.ArgumentParser(description="1:1 Paired Analysis: Original vs Stylized")

    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--stylized_dir", type=str,
                        default="/home/stud/nemmler/retristyle/data/augmented_cache/adain_dino_imagenet_test_r_s71397589",
                        help="Root directory containing stylized image folders")
    parser.add_argument("--view_name", type=str, default="view_001.png",
                        help="Specific view filename inside the sample directory")
    parser.add_argument("--split", type=str, default="test_r")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./results/style_check/styliasation")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    domain_analysis(args)
