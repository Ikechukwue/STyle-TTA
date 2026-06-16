from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
#import nltk
#from nltk.corpus import wordnet as wn
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from .constants import ALL_CLASSIFIERS, ALL_SEEDS, RETRIEVAL_STRATEGIES, N_REFS_VALUES, EVAL_STRATEGIES, TTA_STRATEGIES
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset
from tqdm import tqdm 
from .constants import ALL_CLASSIFIERS
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_json(directory: Path, pattern: str = "*.json") -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))

def get_baseline_results(file_name: str, predictions: bool = True, split: str = 'test_r', 
                         baseline_dir: str = "./results/baseline/tta_inference"):
    classifier = None 
    seed = None
    eval_strat = None 

    for c in ALL_CLASSIFIERS:
        if c in file_name:
            classifier = c
            break
    for s in ALL_SEEDS:
        if str(s) in file_name:
            seed = s  # FIX 1: Assignment corrected from seed = seed
            break
    for es in EVAL_STRATEGIES:
        if es in file_name:
            eval_strat = es
            break
    
    output = "predictions" if predictions else "results"
    output_dir = Path(baseline_dir) / output
    
    # Standard baseline naming pattern
    base_file = output_dir / "imagenet"/ split /  f"{classifier}_geometric_{eval_strat}_nviews1_seed{seed}.json"
    #print(f"Looking for Path:{base_file}")
    return base_file
 
def get_top_k(results: dict, k: int = 5):
    y_pred_matrix = np.array([sample["y_pred"] for sample in results["predictions"]])
    num_samples = y_pred_matrix.shape[0]
    
    k = min(k, y_pred_matrix.shape[1]) # Guard against k > num_classes
    top_k_unsorted_indices = np.argpartition(y_pred_matrix, -k, axis=-1)[:, -k:]
    
    row_indices = np.arange(num_samples)[:, None]
    top_k_unsorted_values = y_pred_matrix[row_indices, top_k_unsorted_indices]
    
    sort_args = np.argsort(-top_k_unsorted_values, axis=-1)
    final_indices = np.take_along_axis(top_k_unsorted_indices, sort_args, axis=-1)
    final_confidences = np.take_along_axis(top_k_unsorted_values, sort_args, axis=-1)
    
    return final_indices, final_confidences

def top_k_acc(results: dict, final_indices: np.ndarray, k: int = 5) -> float:
    y_true = np.array([sample["y_true"] for sample in results["predictions"]])
    top_k_predictions = final_indices[:, :k]
    correct_mask = np.any(top_k_predictions == y_true[:, None], axis=-1)
    return float(np.mean(correct_mask) * 100)

def calc_top_k(pred_path: Path, k: int = 5) -> float:
    if not pred_path.exists():
        return None
    try:
        data = load_json(pred_path)
        final_indices, _ = get_top_k(data, k)
        return top_k_acc(data, final_indices, k)
    except Exception as e:
        print(f"  [error] Could not calculate top-{k} for {pred_path.name}: {e}")
        return None
    
def setup_style():
    """Configure matplotlib for publication-quality plots."""
    if HAS_SNS:
        sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })

def get_names(split:str = "test_r",
                subset_json: str = "./data/imagenet/imagenet_subsets.json",
                name_json:str = "./data/imagenet/imagenet1k/imagenet_class_index.json" ):
    all_ids = load_json(name_json)
    name_dict = {}
    for v in all_ids.values():
        name_dict[v[0]] = v[1]

    split_names = []
    sub_ids = load_json(subset_json)
    split_ids = sub_ids[split]

    split_names = [name_dict[id] for id in split_ids]
    return split_names 



def get_wordnet_taxonomic_order(name_json_path: str = "./data/imagenet/imagenet1k/imagenet_class_index.json") -> list[int]:
    """
    Computes a taxonomic sort order for classes using true WordNet path similarity.
    Groups classes by their lowest common subsumers via hierarchical clustering.
    
    Returns:
        list[int]: A list of class indices sorted by their WordNet hierarchy proximity.
    """
    # Ensure WordNet data is available in the environment
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    # 1. Load the core map to extract WNIDs (e.g., "n02119789")
    try:
        with open(name_json_path) as f:
            class_index_map = json.load(f)
    except Exception as e:
        print(f"Error loading class index map: {e}")
        return []

    # 2. Resolve WNIDs to actual WordNet Synsets
    resolved_synsets = {}
    valid_class_indices = []
    
    for idx_str, (wnid, _) in class_index_map.items():
        idx = int(idx_str)
        try:
            # Parse ImageNet format: n02119789 -> offset 2119789, pos noun ('n')
            offset = int(wnid[1:])
            pos = wnid[0]
            synset = wn.synset_from_pos_and_offset(pos, offset)
            resolved_synsets[idx] = synset
            valid_class_indices.append(idx)
        except Exception:
            # Skip or handle invalid mappings gracefully
            continue

    # Sort indices to establish a deterministic baseline matrix
    valid_class_indices.sort()
    n_classes = len(valid_class_indices)
    
    if n_classes == 0:
        return []

    # 3. Build a Distance Matrix based on WordNet Path Similarity
    # Similarity is bounded (0, 1], where 1.0 means identical synsets.
    # Distance = 1.0 - Similarity
    distance_matrix = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        syn_i = resolved_synsets[valid_class_indices[i]]
        for j in range(i, n_classes):
            syn_j = resolved_synsets[valid_class_indices[j]]
            
            # Compute path similarity (looks at shortest path in hypernym tree)
            sim = syn_i.path_similarity(syn_j)
            if sim is None:
                sim = 0.001 # Fallback minimum connectivity if branches are distinct
                
            dist = 1.0 - sim
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # 4. Perform Hierarchical Clustering to group by lowest mappings
    # Convert square distance matrix to condensed form for scipy linkage
    from scipy.spatial.distance import squareform
    condensed_distances = squareform(distance_matrix)
    
    # Use Ward's minimum variance algorithm to create clean, compact thematic groups
    row_linkage = linkage(condensed_distances, method="ward")
    
    # Extract the optimized leaves order from the tree
    sorted_matrix_indices = leaves_list(row_linkage)
    
    # Map back to your original class integers
    sorted_class_order = [valid_class_indices[i] for i in sorted_matrix_indices]
    
    return sorted_class_order

def normalize_values_inter(data_a: dict, data_b: dict) -> tuple[dict, dict]:
    """
    Finds the global min and max for each metric across to analyse train/split vs baseline test/val 
    (checking both global and class metrics), then scales 'global_summary' and 
    'class_summaries' for both datasets using those shared bounds.
    """
    # 1. Identify all unique base metrics across both files
    base_metrics = set()
    for dataset in [data_a, data_b]:
        # Check global summaries
        for k in dataset.get("global_summary", {}).keys():
            if k.endswith("_mean"):
                base_metrics.add(k.rsplit("_mean", 1)[0])
        # Check class summaries
        for metrics in dataset.get("class_summaries", {}).values():
            for k in metrics.keys():
                if k.endswith("_mean"):
                    base_metrics.add(k.rsplit("_mean", 1)[0])

    # 2. Compute absolute global min/max across both datasets
    global_bounds = {}
    for metric in base_metrics:
        mins = []
        maxs = []
        
        for dataset in [data_a, data_b]:
            # Pull from global summary if min/max keys exist there
            g_sum = dataset.get("global_summary", {})
            if f"{metric}_min" in g_sum: mins.append(g_sum[f"{metric}_min"])
            if f"{metric}_max" in g_sum: maxs.append(g_sum[f"{metric}_max"])
            
            # Pull from all classes
            for metrics in dataset.get("class_summaries", {}).values():
                if f"{metric}_min" in metrics: mins.append(metrics[f"{metric}_min"])
                if f"{metric}_max" in metrics: maxs.append(metrics[f"{metric}_max"])
                
        if mins and maxs:
            global_bounds[metric] = {"min": min(mins), "max": max(maxs)}

    # 3. Scale both datasets (Global and Class-wise summaries)
    for dataset in [data_a, data_b]:
        # --- Scale Global Summary ---
        g_summary = dataset.get("global_summary", {})
        normed_g_means = {}
        for key, val in g_summary.items():
            if key.endswith("_mean"):
                base_metric = key.rsplit("_mean", 1)[0]
                if base_metric in global_bounds:
                    g_min = global_bounds[base_metric]["min"]
                    g_max = global_bounds[base_metric]["max"]
                    denom = g_max - g_min
                    normed_g_means[key] = (val - g_min) / denom if denom != 0 else 0.0
        g_summary.update(normed_g_means)

        # --- Scale Class Summaries ---
        c_summaries = dataset.get("class_summaries", {})
        for class_id, metrics in c_summaries.items():
            normed_c_means = {}
            for key, val in metrics.items():
                if key.endswith("_mean"):
                    base_metric = key.rsplit("_mean", 1)[0]
                    if base_metric in global_bounds:
                        g_min = global_bounds[base_metric]["min"]
                        g_max = global_bounds[base_metric]["max"]
                        denom = g_max - g_min
                        normed_c_means[key] = (val - g_min) / denom if denom != 0 else 0.0
            metrics.update(normed_c_means)
            
    return data_a, data_b

def normalize_values_intra(data: dict) -> dict:
    """
    Finds all metrics ending with '_mean' in both global_summary and class_summaries, 
    and normalizes them in-place [0, 1] using their corresponding internal
    '_min' and '_max' bounds found within their respective dictionaries.
    """
    # --- 1. Normalize Global Summary ---
    global_summary = data.get("global_summary", {})
    normed_global_means = {}
    
    for key, val in global_summary.items():
        if key.endswith("_mean"):
            base_metric = key.rsplit("_mean", 1)[0]
            min_key = f"{base_metric}_min"
            max_key = f"{base_metric}_max"
            
            # Check if this specific dictionary has the min/max bounds
            if min_key in global_summary and max_key in global_summary:
                mean_val = val
                min_val = global_summary[min_key]
                max_val = global_summary[max_key]
                
                denom = max_val - min_val
                normed_global_means[key] = (mean_val - min_val) / denom if denom != 0 else 0.0
                
    # Apply the updates to the global summary block
    global_summary.update(normed_global_means)

    # --- 2. Normalize Class Summaries ---
    class_summaries = data.get("class_summaries", {})
    
    for class_id, metrics in class_summaries.items():
        normed_class_means = {}
        
        for key, val in metrics.items():
            if key.endswith("_mean"):
                base_metric = key.rsplit("_mean", 1)[0]
                min_key = f"{base_metric}_min"
                max_key = f"{base_metric}_max"
                
                if min_key in metrics and max_key in metrics:
                    mean_val = val
                    min_val = metrics[min_key]
                    max_val = metrics[max_key]
                    
                    denom = max_val - min_val
                    normed_class_means[key] = (mean_val - min_val) / denom if denom != 0 else 0.0
        
        # Apply the updates to this specific class dictionary
        metrics.update(normed_class_means)
        
    return data



def get_base_image_folder(dataset: Dataset) -> ImageFolder:
    """Recursively drills down to find the underlying ImageFolder."""
    current_ds = dataset
    while hasattr(current_ds, 'dataset'):
        current_ds = current_ds.dataset
    if not isinstance(current_ds, ImageFolder):
        raise TypeError(f"Expected base dataset to be ImageFolder, but found {type(current_ds)}")
    return current_ds

def inject_stylized_images_inplace(wrapped_dataset, new_base_dir_path="/home/stud/nemmler/retristyle/data/augmented_cache/extracted", view_name="view_001.png"):
    # 1. Drill down to the real ImageFolder
    base_ds = get_base_image_folder(wrapped_dataset)
    
    new_samples = []
    
    # 2. Iterate using enumerate to get the sequential index 'i'
    for i, (old_path, class_idx) in enumerate(base_ds.samples):
        # Format the index to a 5-digit zero-padded folder string (e.g., 0 -> "00000", 12 -> "00012")
        folder_name = f"{i:05d}" 
        
        # Construct the exact path injectively: base_dir / 0000X / view_001.png
        new_path = os.path.join(new_base_dir_path, folder_name, view_name)
        
        # Quick safety check for the very first and last item to ensure paths exist
        if i == 0 or i == len(base_ds.samples) - 1:
            if not os.path.exists(new_path):
                raise FileNotFoundError(f"Generated path does not exist! Check your base directory or padding: {new_path}")
                
        new_samples.append((new_path, class_idx))
        
    # 3. Reassign references inside the base ImageFolder
    base_ds.samples = new_samples
    base_ds.imgs = new_samples 
    
if __name__=="__main__":
    from experiments.data import create_dataset
    from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
    from torchvision.transforms import v2
    import torch 
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=512),
    ])
    ds = create_dataset(
        dataset_name="imagenet",
        data_path="./data",
        split="test_r_c26",
        transform=transform,
    )
    cache_dir = "/home/stud/nemmler/retristyle/data/augmented_cache/extracted"

    inject_stylized_images_inplace(ds, cache_dir)

    print("\n" + "="*50)
    print("DATASET INJECTION VERIFICATION")
    print("="*50)

    # Drill down to print the underlying paths cleanly
    base_ds = ds
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset

    total_samples = len(base_ds.samples)
    first_sample_path, first_label = base_ds.samples[0]
    last_sample_path, last_label = base_ds.samples[-1]

    print(f"Total images in dataset: {total_samples}")
    print(f"First image path:        {first_sample_path} (Class: {first_label})")
    print(f"Last image path:         {last_sample_path} (Class: {last_label})")
    print("="*50)
