"""
Script to extract dataset statistics and update _constants.py
"""

import os
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is .../experiments/data
# we need .../ (the root where experiments module is)
# so we go up 2 levels
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import ast
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torchvision.transforms as transforms

try:
    from code.experiments.data._factory import CustomDataset
    from code.experiments.data import _constants
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser(description="Extract dataset statistics and update _constants.py")
    parser.add_argument("--data_dir", type=str, default="/data/local/colorist/data", help="Path to data directory")
    parser.add_argument("--output_file", type=str, default="/home/staff/sdoerric/research/colorist/experiments/data/_constants.py", help="Path to _constants.py to update")
    return parser.parse_args()


def get_dataset_labels(dataset, dataset_name):
    """
    Attempt to retrieve class label mappings (int -> class_name) from the dataset.
    """
    # 1. MedMNIST (via info dictionary)
    if hasattr(dataset, 'info') and 'label' in dataset.info:
        labels = dataset.info['label']
        if isinstance(labels, dict):
             # Some MedMNIST datasets have nested dicts or just '0': 'name'...
             # Usually it's {'0': 'label0', '1': 'label1'}
             return {int(k): v for k, v in labels.items()}
             
    # Access inner dataset if wrapped
    inner_ds = dataset
    if hasattr(dataset, 'dataset'): # CustomDataset wraps inner dataset
        inner_ds = dataset.dataset

    # 1b. MedMNIST inner
    if hasattr(inner_ds, 'info') and 'label' in inner_ds.info:
        labels = inner_ds.info['label']
        if isinstance(labels, dict):
            return {int(k): v for k, v in labels.items()}
    
    # 2. Peripheral Blood and Bone Marrow Smears (COMMON_CLASSES)
    if hasattr(inner_ds, 'COMMON_CLASSES') and isinstance(inner_ds.COMMON_CLASSES, list):
        # It uses common classes list. List index is usually class index.
        return {i: c for i, c in enumerate(inner_ds.COMMON_CLASSES)}
    
    # 3. Fitzpatrick17k (uses label_to_int)
    if hasattr(inner_ds, 'label_to_int') and isinstance(inner_ds.label_to_int, dict):
        return {v: k for k, v in inner_ds.label_to_int.items()}

    # 4. Standard ImageFolder (has .classes)
    if hasattr(inner_ds, 'classes') and isinstance(inner_ds.classes, list):
        return {i: c for i, c in enumerate(inner_ds.classes)}
    
    # Check if it has class_to_idx
    if hasattr(inner_ds, 'class_to_idx') and isinstance(inner_ds.class_to_idx, dict):
        return {v: k for k, v in inner_ds.class_to_idx.items()}

    # 5. Manual Fallbacks if dynamic lookup fails
    if dataset_name == 'camelyon17wilds':
        return {0: 'background/normal', 1: 'tumor'}
    
    if dataset_name == 'epistr':
        return {0: 'epithelium', 1: 'stroma'} # Often 0=Epi, 1=Stroma, or check
    
    if dataset_name == 'ddi':
        # DDI: 0=Benign, 1=Malignant (usually)
        return {0: 'benign', 1: 'malignant'}
    
    if dataset_name == 'retina':
        return {
            0: 'no DR',
            1: 'mild',
            2: 'moderate',
            3: 'severe',
            4: 'proliferative DR'
        }
        
    if dataset_name == 'pneumoniamnist':
        return {0: 'normal', 1: 'pneumonia'}
        
    if dataset_name == 'breastmnist':
         return {0: 'malignant', 1: 'normal/benign'} # Check docs if unsure

    return {}

def compute_shannon_equitability(incidence_dict, total_classes):
    """
    Compute Shannon's Equitability (E_H).
    E_H = - sum(pi * ln(pi)) / ln(S)
    where pi is proportion of class i, S is total number of classes.
    """
    if not incidence_dict or total_classes <= 1:
        return 0.0 # Undefined or single class
    
    # incidence_dict values sum to 1.0 (approx)
    # Filter out 0 probabilities
    pis = [p for p in incidence_dict.values() if p > 0]
    
    if not pis:
        return 0.0

    shannon_index_H = - sum(p * np.log(p) for p in pis)
    max_H = np.log(total_classes)
    
    if max_H == 0:
        return 0.0
        
    equitability = shannon_index_H / max_H
    return round(equitability, 4)

def compute_statistics_unbatched(dataset, task_type):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    n_samples = 0
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    
    heights = []
    widths = []
    
    class_counts = defaultdict(int)
    
    for imgs, targets in tqdm(loader, desc="Computing stats", leave=False):
        imgs = imgs.double() # Higher precision
        
        batch_size = imgs.shape[0] # Should be 1
        n_samples += batch_size
        
        h, w = imgs.shape[2], imgs.shape[3]
        heights.append(h)
        widths.append(w)
        
        # Mean/Std accumulation
        # imgs: [1, 3, H, W]
        flattened = imgs.view(3, -1)
        channel_sum += flattened.sum(dim=1)
        channel_sq_sum += (flattened ** 2).sum(dim=1)
        pixel_count += (h * w * batch_size)
        
        # Class counts
        if task_type in ["multi-class", "binary-class", "ordinal"]:
            if targets.numel() == 1:
                class_counts[targets.item()] += 1
            else:
                 # Should not happen with batch_size=1 usually
                 pass
        elif task_type == "multi-label":
             t = targets[0]
             for cls_idx, val in enumerate(t):
                 if val > 0:
                     class_counts[cls_idx] += 1

    if pixel_count > 0:
        mean = channel_sum / pixel_count
        mean_sq = channel_sq_sum / pixel_count
        # std = sqrt(E[x^2] - (E[x])^2)
        std = torch.sqrt(torch.clamp(mean_sq - mean ** 2, min=0))
        mean = [round(x, 4) for x in mean.tolist()]
        std = [round(x, 4) for x in std.tolist()]
    else:
        mean = [0.0, 0.0, 0.0]
        std = [0.0, 0.0, 0.0]
    
    if heights:
        h_stats = {
            "min": int(min(heights)),
            "max": int(max(heights)),
            "avg": round(sum(heights) / len(heights), 2)
        }
    else:
        h_stats = {"min": 0, "max": 0, "avg": 0}
        
    if widths:
        w_stats = {
            "min": int(min(widths)),
            "max": int(max(widths)),
            "avg": round(sum(widths) / len(widths), 2)
        }
    else:
        w_stats = {"min": 0, "max": 0, "avg": 0}
    
    # Sort by class index (0, 1, 2...)
    incidence = {k: round(v/n_samples, 4) for k, v in sorted(class_counts.items())} if n_samples > 0 else {}
    
    return {
        "n_samples": n_samples,
        "mean": mean,
        "std": std,
        "height": h_stats,
        "width": w_stats,
        "incidence": incidence
    }

def update_constants_file(filepath, new_data):
    print(f"\nUpdating {filepath}...")
    
    # Generate the string representation
    lines_to_add = []
    lines_to_add.append("\n# ============================================")
    lines_to_add.append("# Detailed Dataset Statistics (Auto-generated)")
    lines_to_add.append("# ============================================")
    
    # Use pprint-like formatting manually to keep it clean
    import json
    
    for name, data in new_data.items():
        lines_to_add.append(f"\n{name} = {{")
        for dataset_key, dataset_val in data.items():
            lines_to_add.append(f"    # {dataset_key}")
            # Format dictionary nicely with indentation
            val_str = json.dumps(dataset_val, indent=4).replace("null", "None")
            # Indent the json string
            val_lines = val_str.split('\n')
            indented_val = val_lines[0] + '\n' + '\n'.join(['    ' + line for line in val_lines[1:]])
            lines_to_add.append(f"    \"{dataset_key}\": {indented_val},")
        lines_to_add.append("}")

    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Check if we already appended stats to avoid duplication
        separator = "# Detailed Dataset Statistics (Auto-generated)"
        if separator in content:
            # Truncate before adding new ones
            content = content.split(separator)[0]
            # Remove trailing newlines/separators
            while content.endswith("\n") or content.endswith("=") or content.endswith("#"):
                content = content[:-1]
            content = content + "\n"
            
        with open(filepath, "w") as f:
            f.write(content + "\n".join(lines_to_add) + "\n")
        
        print("Update complete.")
    except Exception as e:
        print(f"Failed to update file: {e}")

def main():
    args = get_args()
    
    # Prepare result dictionaries
    samples_per_split = {}
    incidence_rates = {}
    normalization_mean_split = {}
    normalization_std_split = {}
    image_dim_stats = {}
    
    # New dictionaries
    class_labels_dict = {}
    shannon_equitability_dict = {}
    
    # We iterate over datasets defined in _constants
    datasets_to_process = _constants.DATASET_SPLITS
    
    print(f"Processing {len(datasets_to_process)} datasets...")
    
    for dataset_name, splits in datasets_to_process.items():
        if dataset_name not in ['retina', 'bone_marrow_smears_and_peripheral_blood']:
            # Skip datasets we already have labels for (to save time)
            print(f"Skipping {dataset_name} (already has labels in _constants.py)")
            continue
        print(f"\nAnalyzing {dataset_name}...")
        
        samples_per_split[dataset_name] = {}
        incidence_rates[dataset_name] = {}
        normalization_mean_split[dataset_name] = {}
        normalization_std_split[dataset_name] = {}
        image_dim_stats[dataset_name] = {}
        shannon_equitability_dict[dataset_name] = {}
        
        task_type = _constants.TASK_TYPE.get(dataset_name, "multi-class")
        total_classes = _constants.NUM_CLASSES.get(dataset_name, 0)
        
        # Try to get class labels from the first split we can load
        labels_found = False
        
        for split in splits:
            print(f"  Split: {split}")
            try:
                ds = CustomDataset(
                    dataset_name=dataset_name, 
                    data_path=args.data_dir, 
                    split=split,
                    transform=transforms.ToTensor(), # Just ToTensor
                    download=True
                )
                
                # Get labels if we haven't yet
                if not labels_found:
                    labels = get_dataset_labels(ds, dataset_name)
                    if labels:
                        class_labels_dict[dataset_name] = labels
                        labels_found = True
                
                stats = compute_statistics_unbatched(ds, task_type)
                
                samples_per_split[dataset_name][split] = stats["n_samples"]
                incidence_rates[dataset_name][split] = stats["incidence"]
                normalization_mean_split[dataset_name][split] = stats["mean"]
                normalization_std_split[dataset_name][split] = stats["std"]
                image_dim_stats[dataset_name][split] = {
                    "height": stats["height"],
                    "width": stats["width"]
                }
                
                # Compute Shannon Equitability
                shannon = compute_shannon_equitability(stats["incidence"], total_classes)
                shannon_equitability_dict[dataset_name][split] = shannon
                
            except Exception as e:
                print(f"  Error processing {dataset_name} {split}: {e}")
                import traceback
                traceback.print_exc()

    # Update _constants.py
    update_constants_file(args.output_file, {
        "CLASS_LABELS": class_labels_dict,
        "SAMPLES_PER_SPLIT": samples_per_split,
        "INCIDENCE_RATES": incidence_rates,
        "SHANNON_EQUITABILITY": shannon_equitability_dict,
        "NORMALIZATION_MEAN_PER_SPLIT": normalization_mean_split,
        "NORMALIZATION_STD_PER_SPLIT": normalization_std_split,
        "IMAGE_DIMENSIONS": image_dim_stats
    })

if __name__ == "__main__":
    main()
