#from experiments.metrics 
from experiments.data import create_dataset
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.metrics.color_metrics import (compute_wasserstein_distance, compute_histogram_distance, 
                               compute_color_moment_distance)
from experiments.metrics.content_metrics import (compute_luminance_ssim, compute_ssim, compute_lpips_distance, compute_edge_similarity)
from experiments.utils.reproducibility import random_seed, worker_seed
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from torchvision.transforms import v2
from pathlib import Path
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
import argparse
from collections import defaultdict

def calculate_singular_metrics(image_a, image_b):

# Compute metrics
    m: Dict[str, float] = {}
    try:
        m["wasserstein"] = compute_wasserstein_distance(image_a, image_b, "rgb")
    except Exception:
        m["wasserstein"] = None
    try:
        kl, js, chi2, hist_int = compute_histogram_distance(image_a, image_b)
        m["histogram_intersection"] = hist_int
    except Exception:
        m["histogram_intersection"] = None
    try:
        m["color_moment_distance"] = compute_color_moment_distance(image_a, image_b)
    except Exception:
        m["color_moment_distance"] = None
    try:
        m["ssim"] = compute_ssim(image_a, image_b)
    except Exception:
        m["ssim"] = None
    try:
        m["luminance_ssim"] = compute_luminance_ssim(image_a, image_b)
    except Exception:
        m["luminance_ssim"] = None
    """   try:
        m["lpips"] = compute_lpips_distance(image_a, image_b)
    except Exception:
        m["lpips"] = None"""
    try:
        m["edge_similarity"] = compute_edge_similarity(image_a, image_b, method="sobel")
    except Exception:
        m["edge_similarity"] = None


    

    return m
 


def group_by_class(dataset, max):
    groups = defaultdict(list)
    
    labels = getattr(dataset, 'targets', None) 
    if labels is None:
        labels = dataset.dataset.dataset.targets 
        
    for i, label in enumerate(labels):
        if len(groups[int(label)]) < max:
            groups[int(label)].append(i)

    return groups

def prepare_datasets(args):

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=256),
    ])
    if args.sub_size > 0:
        set_A = create_dataset(
            args.dataset, args.data_path, args.split,
            transform=transform, use_subset=True, subset_size=args.sub_size, subset_seed=42,
        )
        set_B = create_dataset(
            args.dataset, args.data_path, f"train@{args.split}",
            transform=transform, use_subset=True, subset_size=args.sub_size, subset_seed=42,
        )
    else:
        set_A = create_dataset(
            args.dataset, args.data_path, args.split,
            transform=transform,
        )
        set_B = create_dataset(
            args.dataset, args.data_path, f"train@{args.split}",
            transform=transform,
        )
    
    return set_A, set_B

def domain_analysis(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Datasets
    set_A, set_B = prepare_datasets(args) 

    # 2. Group indices by class
    groups_A = group_by_class(set_A, min(args.max_samples_per_class, 50))
    groups_B = group_by_class(set_B, min(args.max_samples_per_class, 50))

    # Find common classes between both splits
    common_classes = set(groups_A.keys()).intersection(set(groups_B.keys()))
    
    all_metrics = []
    class_summaries = {}
    groups_A_length = sum([len(x) for x in groups_A.values()])
    groups_B_length = sum([len(x) for x in groups_B.values()])
    n_classes = len(common_classes)
    print(f"Found {n_classes}, processing {groups_A_length/n_classes} x {groups_B_length/n_classes} = {(groups_B_length * groups_A_length)/(n_classes*2)}")

    for cls in tqdm(common_classes, desc="Processing Classes"):

        class_metrics = []
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]

        # Nested loop ONLY within the same class
        for img_a in tqdm(images_A, desc="Images"):
            for img_b in images_B:

                
                # Your metric calculation function
                res = calculate_singular_metrics(img_a, img_b)
                class_metrics.append(res)
        
        # Aggregate stats for this specific class
        if class_metrics:
            all_metrics.extend(class_metrics)
            class_summaries[str(cls)] = {
                "wasserstein_mean": np.mean([m['wasserstein'] for m in class_metrics if m['wasserstein'] is not None]),
                "n_pairs": len(class_metrics)
            }

    # 3. Final Global Aggregation
    summary = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            vals = [m[key] for m in all_metrics if m.get(key) is not None]
            if vals:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_std"] = float(np.std(vals))

    # Save detailed class-wise breakdown
    with open(output_dir / f"{args.dataset}_and_{args.split}.json", "w") as f:
        json.dump({
            "global_summary": summary,
            "class_summaries": class_summaries
        }, f, indent=2)

    return summary

def get_args():
    parser = argparse.ArgumentParser(description="Domain Analysis: ImageNet-1k vs ImageNet-1k")

    # --- Data Paths ---
    parser.add_argument("--dataset", type=str, default="imagenet", 
                        help="Dataset name")
    parser.add_argument("--data_path", type=str, default= "./data", 
                        help="Path to the root data directory")
    parser.add_argument("--split", type=str, default="test_r", 
                        help="First split to compare (e.g., train, val)")

    
    # --- Analysis Mode ---
    parser.add_argument("--mode", type=str, choices=["global", "class"], default="class",
                        help="Comparison mode. 'global' compares all, 'class' only compares within same label.")
    parser.add_argument("--sub_size", required=False, type=int, default=0,
                        help="Comparison mode. 'global' compares all, 'class' only compares within same label.")
    
    # --- Complexity Control ---
    parser.add_argument("--max_samples_per_class", type=int, default=30,
                        help="Limit complexity. Set to 0 for no limit.")
    
    parser.add_argument("--output_dir", type=str, default="./results/domain_stats",
                        help="Where to save the JSON results")
    
    # --- Metric Toggles (Optional) ---
    parser.add_argument("--metrics", nargs="+", default=["wasserstein", "histogram", "moments"],
                        help="Specific metrics to run")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    domain_analysis(args)

"""
python -m experiments.thesis.domain_shift
"""
