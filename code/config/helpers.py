from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import os
import re

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import entropy
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .constants import (
    ALL_CLASSIFIERS,
    ALL_SEEDS,
    RETRIEVAL_STRATEGIES,
    N_REFS_VALUES,
    EVAL_STRATEGIES,
    TTA_STRATEGIES,
)
from .paths import DATA_PATH, OUTPUT_PATH

# Optional visualization dependencies
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

# Optional WordNet dependencies
try:
    import nltk
    from nltk.corpus import wordnet as wn
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# ==========================================
# 1. FILE & JSON UTILITIES
# ==========================================

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_json(directory: Path, pattern: str = "*.json") -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def parse_filename_metadata(filepath: str) -> dict:
    """Extract metadata key-value pairs parsed from an experiment prediction filename.

    Returns:
        dict: Extracted metadata with keys (all optional depending on filename matches):
            - 'seed' (int): Random seed value.
            - 'n_refs' (int): Number of reference views/images.
            - 'classifier' (str): Name of the matched classifier.
            - 'eval_strategy' (str): Evaluation strategy matched from EVAL_STRATEGIES.
            - 'tta_method' (str): Method identifier ('geometric_tta', 'style_tta', or 'adain_tta').
    """
    basename = Path(filepath).stem 
    metadata = {}

    seed_match = re.search(r'seed(\d+)', basename)
    if seed_match:
        metadata['seed'] = int(seed_match.group(1))

    nviews_match = re.search(r'nviews(\d+)', basename)
    nrefs_match = re.search(r'nrefs(\d+)', basename)
    n_match = nviews_match if nviews_match else nrefs_match
    if n_match:
        metadata['n_refs'] = int(n_match.group(1))

    for clf in sorted(ALL_CLASSIFIERS, key=len, reverse=True):
        if clf in basename:
            metadata['classifier'] = clf
            break

    for strat in EVAL_STRATEGIES:
        if strat in basename:
            metadata['eval_strategy'] = strat
            break
            
    if 'geometric' in basename:
        metadata['tta_method'] = 'geometric_tta'
    elif 'style_tta' in basename:
        metadata['tta_method'] = 'style_tta'
    elif 'adain' in basename:
        metadata['tta_method'] = 'adain_tta'

    return metadata


def get_prediction_filename(s_key, cfg, ds, cl, ev, retr, rfs, seed, sty=1, use_n=1):
    target_cfg = TTA_STRATEGIES.get(s_key, cfg)
    
    if s_key == "hybrid_tta":
        geo = (rfs - 1) - sty
        return "{ds}_{cl}_hybrid_geo{geo}_sty{sty}_{eval}_split{use_n}_nr{rfs}_seed{seed}_predictions.json".format(
            ds=ds,
            cl=cl,
            geo=f"{geo:02d}",
            sty=f"{sty:02d}",
            eval=ev,
            use_n=use_n,
            rfs=rfs,
            seed=seed,
        )

    fmt_kwargs = {
        "ds": ds,
        "cl": cl, 
        "eval": ev if ev else target_cfg.get("default_eval", "vanilla"), 
        "rfs": rfs,
        "seed": seed, 
        "retr": retr if retr is not None else target_cfg.get("default_retr", ""), 
    }              
    try:
        return target_cfg["template"].format(**fmt_kwargs)
    except KeyError:
        return target_cfg["template"].format(cl=cl, rfs=rfs, seed=seed)


def get_predictions(results: dict):
    """
    Extract true labels and predicted classes from a results JSON.

    Assumes:
        results["predictions"] = [
            {"y_true": ..., "y_pred": [...]},
            ...
        ]

    Returns:
        y_true: shape (N,)
        y_pred: shape (N,)
    """
    y_true = np.array(
        [sample["y_true"] for sample in results["predictions"]]
    )

    y_pred_matrix = np.array(
        [sample["y_pred"] for sample in results["predictions"]]
    )

    y_pred = np.argmax(y_pred_matrix, axis=1)

    return y_true, y_pred


# ==========================================
# 2. METADATA & EXPERIMENT PATH RESOLUTION
# ==========================================

def get_classifier_name(cls: str) -> tuple[str, str]:
    mapping = {
        "resnet18": ("ResNet-18", "CNN"),
        "densenet121": ("DenseNet-121", "CNN"),
        "vit_base_patch16_224": ("ViT-B/16 (224)", "Vision Transformer"),
        "swin_base_patch4_window7_224": ("Swin-B (224)", "Vision Transformer"),
        "ViT-B-16": ("CLIP ViT-B/16", "Vision-Language Model"),
        "ViT-B-16@Zero": ("CLIP ViT-B/16 (Zero-Shot)", "Vision-Language Model"),
        "dinov2_vitb14": ("DINOv2 ViT-B/14", "Foundation Model"),
        "vit_base_patch16_dinov3_lvd1689m": ("DINOv3 ViT-B/16", "Foundation Model"),
    }
    return mapping.get(cls, (cls, "Unknown"))

# Formulas matching the pattern:
# sty = split
# geo = (nr - 1) - sty

def get_hybrid_filename(dataset, model, eval_strat, split, nr, seed):
    sty = split
    geo = (nr - 1) - sty
    
    return f"{dataset}_{model}_hybrid_geo{geo:02d}_sty{sty:02d}_{eval_strat}_split{split}_nr{nr}_seed{seed}_results.json"

def get_names(split: str = "test_r",
              subset_json: str = str(DATA_PATH / "imagenet" / "imagenet_subsets.json"),
              name_json: str = str(DATA_PATH / "imagenet" / "imagenet1k" / "imagenet_class_index.json")) -> List[str]:
    all_ids = load_json(name_json)
    name_dict = {v[0]: v[1] for v in all_ids.values()}

    sub_ids = load_json(subset_json)
    split_ids = sub_ids[split]

    return [name_dict[id] for id in split_ids]

def get_class_name(
    class_idx: int,
    class_names: List[str],
) -> str:
    """Convert a class index into a human-readable ImageNet class name."""
    if class_idx < 0 or class_idx >= len(class_names):
        return f"class {class_idx}"

    return class_names[class_idx]

def get_y_true(dataset: str, split: str) -> list:
    if dataset == "imagenet":
        if split == "test_r":
            labels_data = load_json(
                OUTPUT_PATH / "baseline" / "tta_inference" / "predictions" /
                "imagenet" / "test_r" /
                "densenet121_geometric_tpt_nviews1_seed71397589.json"
            )
            return [i["y_true"] for i in labels_data["predictions"]]
    return []


def get_baseline_results(file_name: str, predictions: bool = True, split: str = 'test_r', dataset="imagenet",
                         baseline_dir: str = "./results/geometric_tta/tta_inference") -> Path:
    classifier = None 
    seed = 71397589
    eval_strat = None 

    for c in ALL_CLASSIFIERS:
        if c in file_name:
            classifier = c
            break

    for es in EVAL_STRATEGIES:
        if es in file_name:
            eval_strat = es
            break
    
    output = "predictions" if predictions else "results"
    output_dir = Path(baseline_dir) / output
    
    return output_dir / dataset / split / f"{classifier}_geometric_{eval_strat}_nviews1_seed{seed}.json"

def baseline_results(cls, dataset, split, prediction=False):
    results = "results" if not prediction else "predictions"
    return str(
        OUTPUT_PATH / "geometric_tta" / "tta_inference" / results /
        dataset / split / f"{cls}_geometric_vanilla_nviews1_seed71397589.json"
    )
# ==========================================
# 3. METRICS & TOP-K EVALUATION
# ==========================================
def extract_class_metrics(pred_path):
    """Extracts accuracy, confidence, and entropy per class from raw predictions."""
    pred_json = load_json(pred_path)
    class_data = {}
    for p in pred_json["predictions"]:
        c_id = str(p["y_true"])
        y_pred_probs = np.array(p["y_pred"])
        
        is_correct = int(np.argmax(y_pred_probs) == p["y_true"])
        confidence = np.max(y_pred_probs)
        pred_entropy = entropy(y_pred_probs)
        
        if c_id not in class_data:
            class_data[c_id] = {"acc": [], "conf": [], "ent": []}
            
        class_data[c_id]["acc"].append(is_correct)
        class_data[c_id]["conf"].append(confidence)
        class_data[c_id]["ent"].append(pred_entropy)
        
    # Average across instances to get clean per-class baseline/TTA values
    return {
        c: {
            "acc": np.mean(v["acc"]),
            "conf": np.mean(v["conf"]),
            "ent": np.mean(v["ent"])
        }
        for c, v in class_data.items()
    }

def get_geometric_mean(pred_path):
    path_data = parse_filename_metadata(pred_path)
    pred_path_obj = Path(pred_path)
    parent_dir = pred_path_obj.parent
    
    seed_matrices = []
    for sd in ALL_SEEDS:
        filename = TTA_STRATEGIES["geometric_tta"]["template"].format(
            cl=path_data["classifier"], 
            eval=path_data["eval_strategy"], 
            rfs=path_data["n_refs"], 
            seed=sd,
            retr=""
        )
        file_path = parent_dir / filename
        if not file_path.exists():
            continue
            
        data = load_json(file_path)
        y_pred_matrix = np.array([sample["y_pred"] for sample in data["predictions"]])
        seed_matrices.append(y_pred_matrix)
    
    if not seed_matrices:
        raise FileNotFoundError("No seed prediction files were found.")
        
    mean_geo = np.mean(seed_matrices, axis=0)
    return mean_geo

def get_top_k(results: dict, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    y_pred_matrix = np.array([sample["y_pred"] for sample in results["predictions"]])
    num_samples, num_classes = y_pred_matrix.shape
    
    k = min(k, num_classes)
    
    if k == 1:
        final_indices = np.argmax(y_pred_matrix, axis=-1)[:, None]
        final_confidences = np.take_along_axis(y_pred_matrix, final_indices, axis=-1)
        return final_indices, final_confidences

    top_k_unsorted_indices = np.argpartition(y_pred_matrix, -k, axis=-1)[:, -k:]
    row_indices = np.arange(num_samples)[:, None]
    top_k_unsorted_values = y_pred_matrix[row_indices, top_k_unsorted_indices]
    
    sort_args = np.argsort(-top_k_unsorted_values, axis=-1)
    final_indices = np.take_along_axis(top_k_unsorted_indices, sort_args, axis=-1)
    final_confidences = np.take_along_axis(top_k_unsorted_values, sort_args, axis=-1)
    
    return final_indices, final_confidences


def top_k_acc(results: dict, final_indices: np.ndarray) -> float:
    y_true = np.array([sample["y_true"] for sample in results["predictions"]])
    correct_mask = np.any(final_indices == y_true[:, None], axis=-1)
    return float(np.mean(correct_mask) * 100)


def calc_top_k(pred_path: Path, k: int = 5) -> Optional[float]:
    if not pred_path.exists():
        return None
    try:
        data = load_json(pred_path)
        final_indices, _ = get_top_k(data, k)
        return top_k_acc(data, final_indices)
    except Exception as e:
        print(f" [error] Could not calculate top-{k} for {pred_path.name}: {e}")
        return None


# ==========================================
# 4. NORMALIZATION UTILITIES
# ==========================================
def latex_escape(text):
        """Escape characters that have special meaning in LaTeX."""
        text = str(text)

        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

def normalize_values_intra(data: dict) -> dict:
    global_summary = data.get("global_summary", {})
    normed_global_means = {}
    
    for key, val in global_summary.items():
        if key.endswith("_mean"):
            base_metric = key.rsplit("_mean", 1)[0]
            min_key, max_key = f"{base_metric}_min", f"{base_metric}_max"
            
            if min_key in global_summary and max_key in global_summary:
                min_val, max_val = global_summary[min_key], global_summary[max_key]
                denom = max_val - min_val
                normed_global_means[key] = (val - min_val) / denom if denom != 0 else 0.0
                
    global_summary.update(normed_global_means)

    class_summaries = data.get("class_summaries", {})
    for class_id, metrics in class_summaries.items():
        normed_class_means = {}
        for key, val in metrics.items():
            if key.endswith("_mean"):
                base_metric = key.rsplit("_mean", 1)[0]
                min_key, max_key = f"{base_metric}_min", f"{base_metric}_max"
                
                if min_key in metrics and max_key in metrics:
                    min_val, max_val = metrics[min_key], metrics[max_key]
                    denom = max_val - min_val
                    normed_class_means[key] = (val - min_val) / denom if denom != 0 else 0.0
                    
        metrics.update(normed_class_means)
        
    return data


def normalize_values_inter(data_a: dict, data_b: dict) -> tuple[dict, dict]:
    base_metrics = set()
    for dataset in [data_a, data_b]:
        for k in dataset.get("global_summary", {}).keys():
            if k.endswith("_mean"):
                base_metrics.add(k.rsplit("_mean", 1)[0])
        for metrics in dataset.get("class_summaries", {}).values():
            for k in metrics.keys():
                if k.endswith("_mean"):
                    base_metrics.add(k.rsplit("_mean", 1)[0])

    global_bounds = {}
    for metric in base_metrics:
        mins, maxs = [], []
        for dataset in [data_a, data_b]:
            g_sum = dataset.get("global_summary", {})
            if f"{metric}_min" in g_sum: mins.append(g_sum[f"{metric}_min"])
            if f"{metric}_max" in g_sum: maxs.append(g_sum[f"{metric}_max"])
            
            for metrics in dataset.get("class_summaries", {}).values():
                if f"{metric}_min" in metrics: mins.append(metrics[f"{metric}_min"])
                if f"{metric}_max" in metrics: maxs.append(metrics[f"{metric}_max"])
                
        if mins and maxs:
            global_bounds[metric] = {"min": min(mins), "max": max(maxs)}

    for dataset in [data_a, data_b]:
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


# ==========================================
# 5. TAXONOMY & WORDNET UTILITIES
# ==========================================

def get_wordnet_taxonomic_order(name_json_path: str = "./data/imagenet/imagenet1k/imagenet_class_index.json") -> list[int]:
    if not HAS_NLTK:
        print("[error] NLTK is required for WordNet operations.")
        return []

    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

    try:
        with open(name_json_path) as f:
            class_index_map = json.load(f)
    except Exception as e:
        print(f"Error loading class index map: {e}")
        return []

    resolved_synsets = {}
    valid_class_indices = []
    
    for idx_str, (wnid, _) in class_index_map.items():
        idx = int(idx_str)
        try:
            offset = int(wnid[1:])
            pos = wnid[0]
            synset = wn.synset_from_pos_and_offset(pos, offset)
            resolved_synsets[idx] = synset
            valid_class_indices.append(idx)
        except Exception:
            continue

    valid_class_indices.sort()
    n_classes = len(valid_class_indices)
    
    if n_classes == 0:
        return []

    distance_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        syn_i = resolved_synsets[valid_class_indices[i]]
        for j in range(i, n_classes):
            syn_j = resolved_synsets[valid_class_indices[j]]
            
            sim = syn_i.path_similarity(syn_j)
            if sim is None:
                sim = 0.001
                
            dist = 1.0 - sim
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    condensed_distances = squareform(distance_matrix)
    row_linkage = linkage(condensed_distances, method="ward")
    sorted_matrix_indices = leaves_list(row_linkage)
    
    return [valid_class_indices[i] for i in sorted_matrix_indices]


# ==========================================
# 6. PLOTTING & VISUALIZATION STYLE
# ==========================================

def setup_style():
    if HAS_SNS:
        sns.set_theme(style="whitegrid", font_scale=1.1)
    if HAS_MPL:
        plt.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
        })


# ==========================================
# 7. PYTORCH DATASET MANIPULATION
# ==========================================
import os
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import ImageFolder


def get_base_image_folder(dataset: Dataset):
    """
    Recursively collects all base datasets (ImageFolder or Camelyon17WILDS)
    unwrapping Subsets and ConcatDatasets.
    """
    current_ds = dataset
    
    # Handle PyTorch Subset unwrapping
    while isinstance(current_ds, Subset):
        current_ds = current_ds.dataset

    if current_ds.__class__.__name__ == 'Camelyon17WILDS' or isinstance(current_ds, ImageFolder):
        return [current_ds]

    # Handle ConcatDataset unwrapping
    if isinstance(current_ds, ConcatDataset):
        base_datasets = []
        for ds in current_ds.datasets:
            base_datasets.extend(get_base_image_folder(ds))
        return base_datasets

    # Generic single-dataset wrapper fallback
    if hasattr(current_ds, 'dataset'):
        return get_base_image_folder(current_ds.dataset)

    raise TypeError(
        f"Expected dataset to wrap ImageFolder or Camelyon17WILDS, but found {type(current_ds)}"
    )


def inject_stylized_images_inplace(
    wrapped_dataset: Dataset, 
    new_base_dir_path: str = str(DATA_PATH / "augmented_cache" / "adain_dino_imagenet_test_r_s71397589"),
    view_name: str = "view_001.png"
):     
    base_datasets = get_base_image_folder(wrapped_dataset)
    
    global_idx = 0
    for base_ds in base_datasets:
        # 1. Standard PyTorch ImageFolder
        if isinstance(base_ds, ImageFolder):
            new_samples = []
            for _, class_idx in base_ds.samples:
                folder_name = f"{global_idx:05d}" 
                new_path = os.path.join(new_base_dir_path, folder_name, view_name)
                
                if global_idx == 0:
                    if not os.path.exists(new_path):
                        raise FileNotFoundError(f"Missing view path: {new_path}")
                        
                new_samples.append((new_path, class_idx))
                global_idx += 1
                
            # Verify the very last path of the full concatenated dataset
            if len(new_samples) > 0:
                last_path = new_samples[-1][0]
                if not os.path.exists(last_path):
                    raise FileNotFoundError(f"Missing view path: {last_path}")

            base_ds.samples = new_samples
            base_ds.imgs = new_samples

        # 2. Camelyon17WILDS Custom Dataset
        elif base_ds.__class__.__name__ == 'Camelyon17WILDS':
            base_ds.inject_stylized_images(
                new_base_dir_path=new_base_dir_path,
                view_name=view_name
            )
            global_idx += len(base_ds)
