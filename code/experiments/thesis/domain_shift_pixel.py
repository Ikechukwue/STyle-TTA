from code.experiments.data import create_dataset
from code.experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from code.experiments.metrics.color_metrics import (compute_wasserstein_distance, compute_histogram_distance,
                               compute_color_moment_distance, batch_wasserstein_distance, batch_color_moment_distance, batch_histogram_distances)
from code.experiments.metrics.content_metrics import (compute_luminance_ssim, compute_ssim, compute_lpips_distance, compute_edge_similarity,
                                                 batch_sobel_edge_similarity, batch_ssim, batch_lpips_distance, _get_lpips_model,)
from code.experiments.utils.reproducibility import random_seed, worker_seed
from code.experiments.metrics.model import model_analysis
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from torchvision.transforms import v2
from pathlib import Path
from code.experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
import argparse
from collections import defaultdict

def _save_json(output_path:Path, results):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

def calculate_batched_metrics(images_A: List[torch.Tensor], images_B: List[torch.Tensor], device="cuda", intra=False):
    """
    Computes all pairwise distances for a class at once using GPU acceleration.
    """
    if not images_A or not images_B:
        return []
    
    batch_A = torch.stack(images_A).to(device)
    batch_B = torch.stack(images_B).to(device)

    with torch.no_grad():

        ssim_matrix = batch_ssim(batch_A, batch_B).cpu().numpy()
        #luminance_matrix = batch_ssim(batch_A, batch_B, use_luminance=True).cpu().numpy()
        edge_matrix = batch_sobel_edge_similarity(batch_A, batch_B).cpu().numpy()
        #wasserstein_matrix = batch_wasserstein_distance(batch_A, batch_B).cpu().numpy()
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
            if i == j and intra:
                batched_pair_results.append({
                    "ssim": None,
                    "edge_similarity": None,
                    #"luminance_ssim": None,
                    #"wasserstein_distance": None,
                    "color_moment_distance": None,
                    "histogramm_distance": None,
                })
                continue
            batched_pair_results.append({
                "ssim": float(ssim_matrix[i, j]),
                "edge_similarity": float(edge_matrix[i, j]),
                #"luminance_ssim": float(luminance_matrix[i, j]),
                #"wasserstein_distance": float(wasserstein_matrix[i,j]),
                "color_moment_distance": float(color_matrix[i,j]),
                "histogramm_distance":float(hist_matrix[i,j]),
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
        # Dataset of the relevant Split 
        set_A = create_dataset(
            args.dataset, args.data_path, "val@test_r", #args.split,
            transform=transform,
        )
        # Dataset of Train-Subset with overlapping classes
        set_B = create_dataset(
            args.dataset, args.data_path, "train@test_r",  #f"train@{args.split}",
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

def classes_analysis(set_A, set_B, args):

    # Group indices by class
    max_samples=30
    max_cls=0

    if args.max_samples:
        max_samples = args.max_samples
    if args.max_classes:
        max_cls = args.max_classes

    intra = args.mode == "intra"
    groups_A = group_by_class(set_A, max_samples, max_cls)

    if intra:
        groups_B = groups_A
        common_classes = set(groups_A.keys())

    else:
        groups_B = group_by_class(set_B, max_samples, max_cls)
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

    models = ["lpips", "ldc", "depthanything_v2_large", "dpt_large" ] # "depthpro", is too big for now and hed is redundant
    model_metrics = model_analysis(set_A, set_B, common_classes, groups_A, groups_B, models, intra)

    for cls in tqdm(common_classes, desc="Processing Classes"):
        images_A = [set_A[i][0] for i in groups_A[cls]]
        images_B = [set_B[i][0] for i in groups_B[cls]]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        class_metrics = calculate_batched_metrics(images_A, images_B, device=device, intra=intra)

        if class_metrics and cls in model_metrics:
            for metric_name, flat_scores in model_metrics[cls].items():
                for idx, pair_dict in enumerate(class_metrics):
                    if idx < len(flat_scores):
                        val = flat_scores[idx]
                        # Safe assignment check: handle model-assigned None values safely
                        pair_dict[metric_name] = float(val) if val is not None else None
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

    return {"global_summary": summary, "class_summaries": class_summaries}


# Main Function
def domain_analysis(args):
    output_dir = Path(args.output_dir)
    result_dir = output_dir / f"{args.dataset}_{args.split}"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_name = f"{args.max_classes}_{args.max_samples}"
    # 1. Prepare Datasets
    # A is of the split (content) and B is from the dataset subset (reference)
    set_A, set_B = prepare_datasets(args) 

    if args.mode == "intra":
        for i, set_ in enumerate([set_A]):#, set_B]):
            results = classes_analysis(set_, set_, args)
            suffix = f"_{args.dataset}_of_{args.split}.json" if i == 1 else f"_{args.split}.json"
            result_name = result_name + suffix
            _save_json(result_dir / result_name, results)
    else:    
        class_results = classes_analysis(set_A, set_B, args)
        result_name += f"_{args.dataset}_{args.split}.json"
        _save_json(result_dir / result_name, class_results)


def get_args():
    parser = argparse.ArgumentParser(description="Domain Analysis: ImageNet-1k vs ImageNet Subsets")

    # --- Data Paths ---
    parser.add_argument("--dataset", type=str, default="imagenet", 
                        help="Dataset name")
    parser.add_argument("--data_path", type=str, default= "./data", 
                        help="Path to the root data directory")
    parser.add_argument("--embedding_path", type=str, default= "./data/embeddings", 
                        help="Path to the root embeddings directory")
    #---- Work Configs ------
    parser.add_argument("--space", type=str, default= "embedding", choices=["pixel", "embedding"], 
                        help="On which of the two levels (embedding or pixel) should it be compared")
    parser.add_argument("--split", type=str, default="test_r", 
                        help="First split to compare (e.g., train, val)")


    
    # --- Analysis Mode ---
    parser.add_argument("--mode", type=str, choices=["intra", "class"], default="class",
                        help="Comparison mode. 'global' compares all, 'class' only compares within same label.")
    
    parser.add_argument("--sub_size", required=False, type=int, default=0,
                        help="Subset size per dataset")
    
    # --- Complexity Control ---
    parser.add_argument("--max_samples", type=int, default=30,
                        help="Limit complexity N² * classes. Set to 0 for no limit.")
    parser.add_argument("--max_classes", type=int, default=0,
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
python -m experiments.thesis.domain_shift --split test_r --max_samples 5 --max_classes 15
"""
