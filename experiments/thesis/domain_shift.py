#from experiments.metrics 
from experiments.data import create_dataset
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.metrics.color_metrics import (compute_wasserstein_distance, compute_histogram_distance, 
                               compute_color_moment_distance, batch_wasserstein_distance, batch_color_moment_distance, batch_histogram_distances)
from experiments.metrics.content_metrics import (compute_luminance_ssim, compute_ssim, compute_lpips_distance, compute_edge_similarity, 
                                                 batch_sobel_edge_similarity, batch_ssim, batch_lpips_distance, _get_lpips_model)
from experiments.utils.reproducibility import random_seed, worker_seed
from experiments.metrics.model import model_analysis
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


def calculate_batched_metrics(images_A: List[torch.Tensor], images_B: List[torch.Tensor], device="cuda"):
    """
    Computes all pairwise distances for a class at once using GPU acceleration.
    """
    if not images_A or not images_B:
        return []
    
    batch_A = torch.stack(images_A).to(device)
    batch_B = torch.stack(images_B).to(device)

    with torch.no_grad():

        ssim_matrix = batch_ssim(batch_A, batch_B).cpu().numpy()
        luminance_matrix = batch_ssim(batch_A, batch_B, use_luminance=True).cpu().numpy()
        edge_matrix = batch_sobel_edge_similarity(batch_A, batch_B).cpu().numpy()
        wasserstein_matrix = batch_wasserstein_distance(batch_A, batch_B).cpu().numpy()
        color_matrix = batch_color_moment_distance(batch_A, batch_B).cpu().numpy() 
        _, _, _, hist_int = batch_histogram_distances(batch_A, batch_B)
        hist_matrix = hist_int.cpu().numpy()
        #features_A = your_feature_extractor(batch_A) # Shape: (30, 512)
        #features_B = your_feature_extractor(batch_B) # Shape: (30, 512)
        
        # Compute all 900 pairwise distances instantly using matrix broadcasting
        # Shape: (30, 30)
        #pairwise_distances = torch.cdist(features_A, features_B, p=2)
    torch.cuda.empty_cache()

    batched_pair_results = []
    N, M = ssim_matrix.shape
    for i in range(N):
        for j in range(M):
            batched_pair_results.append({
                "ssim": float(ssim_matrix[i, j]),
                "edge_similarity": float(edge_matrix[i, j]),
                "luminance_ssim": float(luminance_matrix[i, j]),
                "wasserstein_distance": float(wasserstein_matrix[i,j]),
                "color_moment_distance": float(color_matrix[i,j]),
                "histogramm_distance":float(hist_matrix[i,j]),
            })
    return batched_pair_results

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

def aggregate_class_metrics(class_metrics: list) -> dict:
    """
    Takes a list of individual pair metric dictionaries and computes 
    aggregated statistics (mean, std, min, max) for every metric key found.
    """
    if not class_metrics:
        return {"n_pairs": 0}

    # 1. Dynamically find all metric keys present across the dictionaries
    metric_keys = set()
    for m in class_metrics:
        metric_keys.update(m.keys())
    
    # 2. Compute statistics for each discovered metric
    summary = {}
    for key in metric_keys:
        # Filter out None values safely
        valid_values = [m[key] for m in class_metrics if m.get(key) is not None]
        
        if valid_values:
            summary[f"{key}_mean"] = float(np.mean(valid_values))
            summary[f"{key}_std"] = float(np.std(valid_values))
            summary[f"{key}_min"] = float(np.min(valid_values))
            summary[f"{key}_max"] = float(np.max(valid_values))
        else:
            # Fallback values if all calculations failed for this key
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            summary[f"{key}_min"] = None
            summary[f"{key}_max"] = None

    # 3. Add metadata
    summary["n_pairs"] = len(class_metrics)
    
    return summary

    
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
    
    balanced_groups_A = {}
    balanced_groups_B = {}

    for cls in common_classes:
        count_A = len(groups_A[cls])
        count_B = len(groups_B[cls])
        min_samples = min(count_A, count_B)
        
        if min_samples > 0:
            # Slice the lists down to the exact same size
            balanced_groups_A[cls] = groups_A[cls][:min_samples]
            balanced_groups_B[cls] = groups_B[cls][:min_samples]

    groups_A = balanced_groups_A
    groups_B = balanced_groups_B

    all_metrics = []
    class_summaries = {}
    groups_A_length = sum([len(x) for x in groups_A.values()])
    groups_B_length = sum([len(x) for x in groups_B.values()])
    n_classes = len(common_classes)
    print(f"Found {n_classes}, processing {groups_A_length/n_classes} x {groups_B_length/n_classes} = {(groups_B_length * groups_A_length)/(n_classes*2)}")

    models = ["lpips", "hed", "ldc", "depthanything_v2_large", "dpt_large" ] # "depthpro", is too big for now
    model_metrics = model_analysis(set_A, set_B, common_classes, groups_A, groups_B, models)

    for cls in tqdm(common_classes, desc="Processing Classes"):
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]
        
        # 2. Calculate standard metrics (assuming this returns a list of dicts for each pair)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_metrics = calculate_batched_metrics(images_A, images_B, device=device)

        if class_metrics and cls in model_metrics:
            # 3. model_metrics[cls] is a dict like: {'lpips': array([...]), 'edge_ssim': array([...])}
            # We need to iterate over all metrics calculated for this class
            for metric_name, flat_scores in model_metrics[cls].items():
                
                # 4. Map the flat scores to the pair dictionaries
                # Important: The order in class_metrics must match the order in flat_scores
                for idx, pair_dict in enumerate(class_metrics):
                    if idx < len(flat_scores):
                        pair_dict[metric_name] = float(flat_scores[idx])

            all_metrics.extend(class_metrics)
            class_summaries[str(cls)] = aggregate_class_metrics(class_metrics)


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
                        help="Subset size per dataset")
    
    # --- Complexity Control ---
    parser.add_argument("--max_samples_per_class", type=int, default=30,
                        help="Limit complexity N² * classes. Set to 0 for no limit.")
    
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
python -m experiments.thesis.domain_shift --split test_abl
"""
