import numpy as np
import torch
from .domain_shift_feature import aggregate
from typing import Dict
from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    TASK_TYPE,
    DATASET_SPLITS
)
from torchvision.transforms import v2
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from config.helpers import get_base_image_folder
import json
from PIL import Image
import os

# ============================================================================
# Gram Matrix
# ============================================================================
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized Gram Matrix for a batch of features or a single feature map.
    Handles input shapes of (B, C, H, W) or (C, H, W).
    """
    if feat.dim() == 4:
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2))
        return gram / (c * h * w)
    elif feat.dim() == 3:
        c, h, w = feat.shape
        f = feat.view(c, h * w)
        return torch.mm(f, f.t()) / (c * h * w)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {feat.dim()}D")

def build_vgg_extractor(layer: int = 18) -> torch.nn.Module:
    """relu3_2 by default — captures mid-level texture without semantics."""
    import torchvision.models as models
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:layer]
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg

def gram_distance_batch(
    imgs_A: torch.Tensor,  # (N, 3, H, W) normalized
    imgs_B: torch.Tensor,  # (M, 3, H, W) normalized
    extractor: torch.nn.Module,
    device: str = "cpu",
) -> float:
    """Mean pairwise Gram MSE between two sets of images."""
    extractor = extractor.to(device)
    
    with torch.no_grad():
        feats_A = extractor(imgs_A.to(device))  # (N, C, H', W')
        feats_B = extractor(imgs_B.to(device))  # (M, C, H', W')
    
    grams_A = gram_matrix(feats_A)  # (N, C, C)
    grams_B = gram_matrix(feats_B)  # (M, C, C)
    
    mean_gram_A = grams_A.mean(dim=0)
    mean_gram_B = grams_B.mean(dim=0)
    
    return float(torch.nn.functional.mse_loss(mean_gram_A, mean_gram_B).item())

def create_gram_results(dataset, split, data_path, limit_classes=None):
    print("\n" + "="*50)
    print(f"INITIALIZING GRAM ANALYSIS | Dataset: {dataset} | Split: {split}")
    print("="*50)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=256, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop(size=224),
        v2.Normalize(mean=NORMALIZATION_MEAN[dataset], std=NORMALIZATION_STD[dataset])
    ])

    print("Loading dataset manifests...")
    set_a = create_dataset(dataset_name=dataset, data_path=data_path, split=split, transform=transform)
    set_train = create_dataset(dataset_name=dataset, data_path=data_path, split=f"train@{split}", transform=transform)
    set_val = create_dataset(dataset_name=dataset, data_path=data_path, split=f"val@{split}", transform=transform)

    def group_by_class(dataset_obj, limit=None):
        images = get_base_image_folder(dataset_obj)
        samples = np.array([s[0] for s in images.samples]) 
        labels = np.array(images.targets)
        all_labels = np.unique(labels)
        if limit is not None:
            all_labels = all_labels[:limit]

        groups = {}
        for cls in all_labels:
            groups[int(cls)] = samples[labels == cls].tolist()
        return groups

    def paths_to_tensor(paths, img_transform):
        tensors = []
        for p in paths:
            with Image.open(p).convert("RGB") as img:
                tensors.append(img_transform(img))
        return torch.stack(tensors)

    def compute_class_metrics(group_A, group_B, desc_string="Processing"):
        common = sorted(set(group_A.keys()) & set(group_B.keys()))
        results = {}
        model = build_vgg_extractor()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from tqdm import tqdm

        for cls in tqdm(common, desc=desc_string):
            paths_X = group_A[cls]
            paths_Y = group_B[cls]
            
            n = min(len(paths_X), len(paths_Y))
            if n == 0:
                continue
                
            X = paths_to_tensor(paths_X[:n], transform)
            Y = paths_to_tensor(paths_Y[:n], transform)
            
            results[cls] = {
                "gram_distance": gram_distance_batch(X, Y, model, device),
                "n_samples":   n,
            }
        return results

    def compute_net_shift(
        domain_cls: Dict[int, Dict[str, float]],
        baseline_cls: Dict[int, Dict[str, float]],
    ) -> Dict[int, Dict[str, float]]:
        common = sorted(set(domain_cls.keys()) & set(baseline_cls.keys()))
        metrics = ["gram_distance"]
        results = {}
        for cls in common:
            results[cls] = {
                f"{m}_net": domain_cls[cls][m] - baseline_cls[cls][m]
                for m in metrics
                if m in domain_cls[cls] and m in baseline_cls[cls]
            }
        return results
    
    print("Structuring sample groups by classes...")
    groups_domain = group_by_class(set_a, limit=limit_classes)
    groups_train = group_by_class(set_train, limit=limit_classes)
    groups_val = group_by_class(set_val, limit=limit_classes)

    print("\nComputing Distance Arrays...")
    baseline_results = compute_class_metrics(groups_val, groups_train, desc_string="In-Dist Gap (Val vs Train)")
    domain_results = compute_class_metrics(groups_domain, groups_train, desc_string="Domain Gap  (Shift vs Train)")

    net_shift_cls = compute_net_shift(domain_results, baseline_results)

    results = {
        "splits": {
            "domain":   split,
            "train":    f"train@{split}",
            "val":      f"val@{split}",
        },
        "domain_gap": {
            "global":    aggregate(domain_results),
            "per_class": {str(k): v for k, v in domain_results.items()},
        },
        "baseline_gap": {
            "global":    aggregate(baseline_results),
            "per_class": {str(k): v for k, v in baseline_results.items()},
        },
        "net_shift": {
            "global":    aggregate({k: v for k, v in net_shift_cls.items()}),
            "per_class": {str(k): v for k, v in net_shift_cls.items()},
        },
    }
    return results

if __name__ == "__main__":
    dataset = "imagenet"
    split = "test_r"
    data_path = "./data"
    
    # Set to None to process all classes in the dataset.
    MAX_CLASSES = None
    
    gram_res = create_gram_results(dataset, split, data_path, limit_classes=MAX_CLASSES)

    if gram_res:
        # ====================================================================
        # Final Output Summary Table
        # ====================================================================
        print("\n" + "="*70)
        print("METRIC METADATA SUMMARY")
        print("="*70)
        print(f"Domain Setup:  {gram_res['splits']['domain']}")
        print(f"Baseline Val:  {gram_res['splits']['val']}")
        print("-"*70)
        print(f"{'Class ID':<10} | {'Samples':<10} | {'Baseline Distance':<18} | {'Domain Distance':<16} | {'Net Shift':<12}")
        print("-"*70)
        
        per_class_domain = gram_res["domain_gap"]["per_class"]
        per_class_base = gram_res["baseline_gap"]["per_class"]
        per_class_net = gram_res["net_shift"]["per_class"]
        
        for cls_id in sorted(per_class_domain.keys()):
            n_samples = per_class_domain[cls_id]["n_samples"]
            b_dist = per_class_base.get(cls_id, {}).get("gram_distance", 0.0)
            d_dist = per_class_domain[cls_id]["gram_distance"]
            net = per_class_net.get(cls_id, {}).get("gram_distance_net", 0.0)
            print(f"{cls_id:<10} | {n_samples:<10} | {b_dist:<18.5f} | {d_dist:<16.5f} | {net:<12.5f}")
            
        print(
            f"{'GLOBAL AGG':<10} | "
            f"{'-':<10} | "
            f"{gram_res['baseline_gap']['global']['gram_distance_mean']:<18.5f} | "
            f"{gram_res['domain_gap']['global']['gram_distance_mean']:<16.5f} | "
            f"{gram_res['net_shift']['global']['gram_distance_net_mean']:<12.5f}"
        )

        # Save Configuration
        output_dir = "./results/domain_gap"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "gram_summary.json")
        
        with open(output_path, "w") as f:
            json.dump(gram_res, f, indent=2)
        print(f"File archived safely to: {output_path}\n")
