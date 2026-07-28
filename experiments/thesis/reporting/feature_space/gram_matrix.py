import numpy as np
import torch
import torch.nn.functional as F
from .domain_shift_feature import aggregate
from typing import Dict, List, Optional, Tuple, Generator
from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    TASK_TYPE,
    DATASET_SPLITS
)
from torchvision.transforms import v2
from config.helpers import get_base_image_folder
import json
from PIL import Image
import os
from tqdm import tqdm

# ============================================================================
# Core Math & Network Extraction
# ============================================================================
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Computes the normalized Gram Matrix for a batch of features (B, C, H, W)."""
    if feat.dim() == 4:
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (c * h * w)
    else:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got {feat.dim()}D")

def build_vgg_extractor(layer: int = 18) -> torch.nn.Module:
    import torchvision.models as models
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:layer]
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg

# ============================================================================
# Memory-Safe Data Loading & Path Resolution Helpers
# ============================================================================
def get_class_metadata_map(dataset_obj, limit_classes: Optional[int] = None) -> Dict[int, List[Tuple[str, int]]]:
    """
    Groups dataset samples by class index while retaining the original global sequence index 'i'.
    Returns: { class_idx: [(old_path, global_i), ...] }
    """
    base_ds = get_base_image_folder(dataset_obj)
    groups = {}
    
    for i, (old_path, class_idx) in enumerate(base_ds.samples):
        cls_id = int(class_idx)
        if limit_classes is not None and cls_id >= limit_classes:
            continue
        if cls_id not in groups:
            groups[cls_id] = []
        groups[cls_id].append((old_path, i))
        
    return groups

def use_style_image(
    metadata_chunk: List[Tuple[str, int]], 
    k: int, 
    transform: v2.Compose, 
    augmented_dir: str
) -> torch.Tensor:
    """
    Resolves the original/stylized paths for a small chunk of items up to view 'k',
    loads them, applies transforms, and returns a single combined tensor batch.
    """
    tensors = []
    for old_path, global_i in metadata_chunk:
        folder_name = f"{global_i:05d}"
        if k == 0:
            # View 0 is always the clean original non-stylized image
            img_path = old_path
        else: 
            view_name = f"view_{k:03d}.png"
            img_path = os.path.join(augmented_dir, folder_name, view_name)
            
        with Image.open(img_path).convert("RGB") as img:
            tensors.append(transform(img))
    return torch.stack(tensors)

# ============================================================================
# Incremental Feature Averaging Engine
# ============================================================================
def compute_incremental_class_gram(
    metadata_items: List[Tuple[str, int]],
    k: int,
    transform: v2.Compose,
    extractor: torch.nn.Module,
    device: str,
    augmented_dir: str,
    chunk_size: int = 16
) -> torch.Tensor:
    """
    Streams a class's items in small chunks through VGG, computes individual Gram
    matrices, and aggregates a running sum on the GPU to remain completely OOM-safe.
    """
    running_gram_sum = None
    total_images_processed = 0
    
    # Process the class tracking list in small bite-sized chunks
    for start_idx in range(0, len(metadata_items), chunk_size):
        chunk = metadata_items[start_idx:start_idx + chunk_size]
        
        # Resolves paths to style images with k != 0
        X_batch = use_style_image(chunk, k, transform, augmented_dir)
        
        with torch.no_grad():
            # Extract features: Shape (B * k, C, H', W')
            feats = extractor(X_batch.to(device))
            # Calculate individual Gram Matrices: Shape (B * k, C, C)
            grams = gram_matrix(feats)
            
            # Sum up Gram configurations across the current batch dimension
            batch_gram_sum = grams.sum(dim=0)
            
            if running_gram_sum is None:
                running_gram_sum = batch_gram_sum
            else:
                running_gram_sum += batch_gram_sum
                
            total_images_processed += X_batch.shape[0]
            
    # Calculate global mean Gram matrix for this specific class distribution
    return (running_gram_sum / total_images_processed).cpu()

# ============================================================================
# Orchestrator Workflow
# ============================================================================
def create_gram_results(
    dataset: str, 
    split: str, 
    data_path: str, 
    augmented_cache_dir: str,
    limit_classes: Optional[int] = None, 
    use_stylized: List[int] = [1],
    base_chunk_size: int = 16
) -> Dict[str, dict]:
    
    print("\n" + "="*60)
    print(f"INITIALIZING MEMORY-SAFE GRAM ANALYSIS")
    print(f"Dataset: {dataset} | Target Split: {split} | K-Intervals: {use_stylized}")
    print("="*60)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=256, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(size=224),
        v2.Normalize(mean=NORMALIZATION_MEAN[dataset], std=NORMALIZATION_STD[dataset])
    ])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_vgg_extractor().to(device)
    
    
    # --- Step 1: Compute Stable Baselines (Train & Val) ---
    print("\n>>> Extracting Stable Reference Distributions (Train vs Val)...")
    ds_train = create_dataset(dataset_name=dataset, data_path=data_path, split=f"train@{split}", transform=transform)
    ds_val = create_dataset(dataset_name=dataset, data_path=data_path, split=f"val@{split}", transform=transform)
    
    train_meta = get_class_metadata_map(ds_train, limit_classes)
    val_meta = get_class_metadata_map(ds_val, limit_classes)
    
    common_baseline_classes = sorted(set(train_meta.keys()) & set(val_meta.keys()))
    
    train_grams = {}
    val_grams = {}
    
    for cls in tqdm(common_baseline_classes, desc="Processing Base Baselines"):
        # For standard sets, we use k=1 (only original images) and pass an empty cache path string
        train_grams[cls] = compute_incremental_class_gram(train_meta[cls], 0, transform, model, device, "", base_chunk_size)
        val_grams[cls] = compute_incremental_class_gram(val_meta[cls], 0, transform, model, device, "", base_chunk_size)

    # --- Step 2: Loop Over Target K-views for Domain Shifts ---
    print("\n>>> Extracting Multi-View Domain Shifts...")
    ds_domain = create_dataset(dataset_name=dataset, data_path=data_path, split=split, transform=transform)
    domain_meta = get_class_metadata_map(ds_domain, limit_classes)
    
    final_archive = {}
    
    for k in use_stylized:
        out_split_name = f"{split}_k{k}"
        print(f"\nEvaluating slice context: {out_split_name} ({k} view image)")
        
        domain_gap_cls = {}
        baseline_gap_cls = {}
        net_shift_cls = {}
        
        # Process class arrays matching across all data manifests
        eval_classes = sorted(set(domain_meta.keys()) & set(train_grams.keys()))
        
        for cls in tqdm(eval_classes, desc=f"Computing Metrics for K={k}"):
            # Dynamically compute domain representation over K views safely
            domain_gram_cls = compute_incremental_class_gram(
                domain_meta[cls], k, transform, model, device, augmented_cache_dir, base_chunk_size
            )
            
            t_gram = train_grams[cls]
            v_gram = val_grams[cls]
            
            # Metric evaluation via MSE differences between global class averages
            d_dist = float(torch.nn.functional.mse_loss(domain_gram_cls, t_gram).item())
            b_dist = float(torch.nn.functional.mse_loss(v_gram, t_gram).item())
            
            n_samples = len(domain_meta[cls])
            
            domain_gap_cls[cls] = {"gram_distance": d_dist, "n_samples": n_samples}
            baseline_gap_cls[cls] = {"gram_distance": b_dist, "n_samples": n_samples}
            net_shift_cls[cls] = {"gram_distance_net": d_dist - b_dist, "n_samples": n_samples}
            
        final_archive[out_split_name] = {
            "splits": {
                "domain":   out_split_name,
                "train":    f"train@{split}",
                "val":      f"val@{split}",
            },
            "domain_gap": {
                "global":    aggregate(domain_gap_cls),
                "per_class": {str(key): val for key, val in domain_gap_cls.items()},
            },
            "baseline_gap": {
                "global":    aggregate(baseline_gap_cls),
                "per_class": {str(key): val for key, val in baseline_gap_cls.items()},
            },
            "net_shift": {
                "global":    aggregate(net_shift_cls),
                "per_class": {str(key): val for key, val in net_shift_cls.items()},
            },
        }
        
    return final_archive

if __name__ == "__main__":
    DATASET = "imagenet"
    SPLIT = "test_r"
    DATA_PATH = "./data"
    CACHE_DIR = "/home/stud/nemmler/retristyle/data/augmented_cache/dino_imagenet_test_r_s71397589"
    
    K_INTERVALS = [0,1,2,3,4]
    MAX_CLASSES = None  # Change to an integer if you want a fast validation subset test
    
    gram_res_bundle = create_gram_results(
        dataset=DATASET,
        split=SPLIT,
        data_path=DATA_PATH,
        augmented_cache_dir=CACHE_DIR,
        limit_classes=MAX_CLASSES,
        use_stylized=K_INTERVALS,
        base_chunk_size=16  # Controls max VGG activations footprint on your GPU
    )

    # Save Bundle to JSON archive
    output_dir = "./results/domain_gap"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gram_stylized_summary.json")
    
    with open(output_path, "w") as f:
        json.dump(gram_res_bundle, f, indent=2)
    print(f"\n[Success] All metrics evaluated and archived safely to: {output_path}")
