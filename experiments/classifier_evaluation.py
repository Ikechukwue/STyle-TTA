"""
xAILab Bamberg
University of Bamberg

@description:
Classifier evaluation script for trained models using various augmentation/DG methods.

Evaluates trained classifiers and reports:
- Accuracy
- Balanced Accuracy
- AUC-ROC
- Expected Calibration Error (ECE)

Supports evaluation on train, val, and test splits.
"""

import json
import timm
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Dict
from torch import Generator
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from accelerate import Accelerator
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
# Import project modules
from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    TASK_TYPE,
    DATASET_SPLITS
)
from experiments.tta.checkpoint import (
    build_experiment_key,
    predictions_path,
    results_path,
    load_predictions,
    save_predictions,
    save_result,
)
from experiments.utils.reproducibility import random_seed, worker_seed
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio


# =============================================================================
# Constants
# =============================================================================
DEFAULT_SEED = 265017005

# Method categories for reference
BASELINE_AUGMENTATIONS = [
    "aug_mix", "auto_augment", "color_jitter", "gray_scale", "rand_augment",
    "random_erasing", "random_flip", "random_resized_crop", "targeted_augment",
    "trivial_augment", "none"
]

COLOR_TRANSFER_METHODS = [
    "adain", "adaattn", "artflow", "efdm", "iecontrast", "mast", "sanet",
    "styleformer", "stytr2", "photonas", "deeppreset", "modflows", "wct2",
    "contrimix", "sgvits", "stylizing_vit"
]

DOMAINBED_METHODS = [
    "ERM", "IB_ERM", "Mixup", "RSC", "SD", "SelfReg"
]

SDG_METHODS = [
    "JiGen", "ADA", "MEADA", "L2D", "ADVST", "ACVC", "CCSA", "RSDA",
    "PDEN", "MMLD", "MixStyle", "StyDeSty"
]

# Colorist color transfer configuration
COLORIST_COLOR_SPACES = [
    "rgb", "lab", "hed", "hsv", "hsi", "hsd", "lch", "luv",
    "yuv", "ycbcr", "yiq", "ypbpr", "ydbdr"
]

COLORIST_STRATEGIES = ["direct", "regional"]

COLORIST_MATCHING = "mean_std"

AUGMENTATION_PERCENTAGES = [
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50
]


# =============================================================================
# Calibration Metrics
# =============================================================================
def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    task_type: str,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy,
    weighted by the proportion of samples in each confidence bin.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities (softmax/sigmoid output)
        num_classes: Number of classes
        task_type: Task type ("multi-class", "multi-label", etc.)
        n_bins: Number of bins for calibration (default: 15)
        
    Returns:
        ECE value (lower is better, 0 is perfectly calibrated)
    """
    if task_type == "multi-label":
        # For multi-label, compute ECE per label and average
        n_labels = y_pred.shape[1]
        ece_per_label = []
        
        for i in range(n_labels):
            probs = y_pred[:, i]
            labels = y_true[:, i]
            ece_i = _compute_binary_ece(labels, probs, n_bins)
            ece_per_label.append(ece_i)
        
        return float(np.mean(ece_per_label))
    
    elif num_classes == 2 or task_type in ["binary-class"]:
        # Binary classification - use probability of positive class
        probs = y_pred[:, -1] if y_pred.ndim > 1 else y_pred
        labels = y_true.squeeze()
        return _compute_binary_ece(labels, probs, n_bins)
    
    else:
        # Multi-class classification
        return _compute_multiclass_ece(y_true.squeeze(), y_pred, n_bins)


def _compute_binary_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute ECE for binary classification.
    
    Args:
        y_true: Binary ground truth labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins
        
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        # Find samples in this bin
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.sum(in_bin) / total_samples
        
        if prop_in_bin > 0:
            # Average confidence in bin
            avg_confidence = np.mean(y_prob[in_bin])
            # Average accuracy in bin
            avg_accuracy = np.mean(y_true[in_bin])
            # Add to ECE
            ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)
    
    return float(ece)


def _compute_multiclass_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute ECE for multi-class classification.

    Uses the maximum predicted probability (confidence) for each sample.

    Args:
        y_true: Ground truth class labels
        y_pred: Predicted probabilities for all classes
        n_bins: Number of bins

    Returns:
        ECE value
    """
    # Get predicted class and confidence
    confidences = np.max(y_pred, axis=1)
    predictions = np.argmax(y_pred, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.sum(in_bin) / total_samples
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += prop_in_bin * np.abs(avg_accuracy - avg_confidence)
    
    return float(ece)


# =============================================================================
# Classification Metrics
# =============================================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    task_type: str
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities (softmax/sigmoid output)
        num_classes: Number of classes
        task_type: Task type ("multi-class", "multi-label", etc.)

    Returns:
        Dictionary with accuracy, balanced_accuracy, auc, and ece
    """
    # Compute ECE first (uses probabilities directly)
    ece = compute_ece(y_true, y_pred, num_classes, task_type)
    
    if task_type == "multi-label":
        # Multi-label classification
        y_pred_labels = (y_pred > 0.5).astype(int)
        n_labels = y_true.shape[1]

        acc_sum = 0.0
        for i in range(n_labels):
            acc_sum += accuracy_score(y_true[:, i], y_pred_labels[:, i])
        accuracy = acc_sum / n_labels
        balanced_acc = accuracy  # Not applicable for multi-label

        try:
            auc_sum = 0.0
            for i in range(n_labels):
                auc_sum += roc_auc_score(y_true[:, i], y_pred[:, i])
            auc = auc_sum / n_labels
        except ValueError:
            auc = 0.0 
            
    elif num_classes == 2 or task_type in ["binary-class"]:
        # Binary classification
        y_true_squeezed = y_true.squeeze()
        y_pred_labels = (y_pred[:, -1] > 0.5).astype(int)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)
        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        top_5_accuracy = accuracy
        try:
            auc = roc_auc_score(y_true_squeezed, y_pred[:, -1])
        except ValueError:
            auc = 0.0
            
    else:
        # Multi-class classification
        y_true_squeezed = y_true.squeeze()
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)

        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)
        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        
        k = min(5, num_classes)
        top_k_indices = np.argpartition(y_pred, -k, axis=1)[:, -k:]
        
        match_mask = top_k_indices == y_true_squeezed[:, None]
        top_5_accuracy = np.any(match_mask, axis=1).mean()

        try:
            if len(y_true_squeezed) > 30000:
                print(f"  [Note] Dataset large ({len(y_true_squeezed)} samples). Stratifying 30k items for AUC...")
                _, y_true_sub, _, y_pred_sub = train_test_split(
                    y_true_squeezed, 
                    y_pred, 
                    test_size=30000,
                    stratify=y_true_squeezed,   
                    random_state=42
                )
            else:
                y_true_sub = y_true_squeezed
                y_pred_sub = y_pred            
            active_classes = np.sort(np.unique(y_true_sub))
            
            y_pred_sliced = y_pred_sub[:, active_classes]
            
            row_sums = y_pred_sliced.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-9, row_sums)
            y_pred_sliced = y_pred_sliced / row_sums
            
            auc = roc_auc_score(
                y_true_sub, 
                y_pred_sliced, 
                multi_class="ovr", 
                labels=active_classes
            )

        except Exception as e:
            print(f"  [Warning] Global AUC calculation fallback triggered: {e}")
            auc = 0.0

    return {
        'accuracy': float(accuracy),
        'top5_accuracy': float(top_5_accuracy),
        'balanced_accuracy': float(balanced_acc),
        'auc': float(auc),
        'ece': float(ece)
    }

# =============================================================================
# Data Loading
# =============================================================================
def prepare_dataloader(
    dataset: str,
    data_path: str,
    split: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    g: Generator,
    classifier: Optional[str] = None,
) -> DataLoader:
    """
    Prepare a dataloader for evaluation.
    
    Args:
        dataset: Dataset name
        data_path: Path to dataset
        split: Data split ('train', 'val', 'test')
        input_size: Input size for model
        batch_size: Batch size
        num_workers: Number of dataloader workers
        g: Random generator for reproducibility
        
    Returns:
        DataLoader for the specified split
    """
    # Standard evaluation transform
    stats_name = dataset if not "ViT" in classifier else "ViT"

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=input_size),
        v2.Normalize(mean=NORMALIZATION_MEAN[stats_name], std=NORMALIZATION_STD[stats_name])
    ])
    
    # Load dataset
    eval_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split=split,
        transform=transform,
    )
    
    # Create dataloader
    eval_loader = DataLoader(
        dataset=eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
        generator=g,
    )
    
    return eval_loader


# =============================================================================
# CLIP / DINOv2 classifier names
# =============================================================================
_CLIP_CLASSIFIERS = {"ViT-B-16", "ViT-L-14", "ViT-B-32", "ViT-B-16@Zero"}
_DINOV2_CLASSIFIERS = {"dinov2_vitb14", "dinov2_vitl14", "dinov2_vits14", "dinov2_vitg14", "vit_base_patch16_dinov3_lvd1689m"}


# =============================================================================
# Model Loading
# =============================================================================
class MaskedClassifier(nn.Module):
    """Wrapper Class to have the models predictions be masked for a subset of classes"""
    def __init__(self, base_model, split):
        super().__init__()
        self.base_model = base_model
        self.split = split
        mask_list = self._set_class_mask()
        self.register_buffer("mask", torch.tensor(mask_list, dtype=torch.bool))

        if hasattr(base_model, 'backbone'):
            self.backbone = base_model.backbone
        else:
            # Fallback for standard models where the model IS the backbone
            self.backbone = base_model

    def forward(self, x):
        return self.base_model(x)[:, self.mask]
    

    @property
    def head(self):
        return self.base_model.head
    
    def _set_class_mask(self) -> list[bool]:
        """
        Create subset mask for the wnids that have to be ignored for correct softmax calculation.

        A json of all subset (as well as the all ImageNet classes) wnids is required
        """

        with open("./data/imagenet/imagenet_subsets.json", "r") as f:
            data = json.load(f)
            if "@" in self.split:
                split = self.split.split("@")[1]
            else:
                split = self.split 
            subset_wnids = data.get(split, [])
            all_wnids = data.get("full", [])

        mask = [wnid in subset_wnids for wnid in all_wnids]

        return mask




def load_classifier(
    weights_path: str,
    classifier: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Load a trained classifier model.
    
    Args:
        weights_path: Path to model weights
        classifier: Classifier architecture name (from timm, open_clip, or DINOv2)
        num_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    # ---- CLIP models (via open_clip) ----------------------------------------
    if classifier in _CLIP_CLASSIFIERS:
        from experiments.clip_classifier import load_clip_classifier
        mode = "linear_probe" if weights_path and (Path(weights_path).exists() or weights_path=="linear_probe") else "zero_shot"
        return load_clip_classifier(
            model_name=classifier,
            num_classes=num_classes,
            device=str(device),
            weights_path=weights_path if mode == "linear_probe" else None,
            mode=mode,
        )

    # ---- DINOv2 models (via torch.hub) --------------------------------------
    if classifier in _DINOV2_CLASSIFIERS:
        from experiments.clip_classifier import load_dino_classifier
        return load_dino_classifier(
            model_name=classifier,
            num_classes=num_classes,
            device=str(device),
            weights_path=weights_path if weights_path and Path(weights_path).exists() else None,
        )

    # ---- Standard timm models -----------------------------------------------
    # Create model architecture
    if weights_path == "pretrained":
        model = timm.create_model(classifier, pretrained=True, num_classes=num_classes)
        return model
    
    model = timm.create_model(classifier, pretrained=False, num_classes=num_classes)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Handle different state dict formats
    if isinstance(state_dict, dict):
        if 'model' in state_dict and isinstance(state_dict.get('model'), dict):
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict and isinstance(
            state_dict.get('state_dict'), dict
        ):
            state_dict = state_dict['state_dict']

    # Remove 'module.' prefix if present (from DataParallel/DDP)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    # Compare loaded keys against model keys to detect prefix issues
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(new_state_dict.keys())
    overlap = model_keys & loaded_keys

    if len(overlap) < len(model_keys) * 0.5 and loaded_keys:
        # Keys don't match — try auto-remapping common DomainBed prefixes
        print(f"\n  Key mismatch: {len(overlap)}/{len(model_keys)} overlap.")
        print(f"  Loaded sample: {sorted(loaded_keys)[:3]}")
        print(f"  Model sample:  {sorted(model_keys)[:3]}")
        remapped = _try_remap_state_dict(new_state_dict, model.state_dict())
        if remapped is not None:
            new_state_dict = remapped
            print(f"  Auto-remapped to {len(new_state_dict)} keys.")

    # Fail fast on empty state dicts instead of silently using random weights
    if not new_state_dict:
        raise RuntimeError(
            f"Loaded state dict from {weights_path} is empty. "
            f"The model weights file may be corrupt or was saved incorrectly."
        )

    # Load state dict
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError:
        result = model.load_state_dict(new_state_dict, strict=False)
        n_loaded = len(model_keys) - len(result.missing_keys)
        if n_loaded == 0:
            raise RuntimeError(
                f"No weights could be loaded from {weights_path}. "
                f"Loaded keys: {sorted(new_state_dict.keys())[:5]}, "
                f"Expected keys: {sorted(model_keys)[:5]}"
            )
        print(f"\n  \u26a0 Non-strict load: {n_loaded}/{len(model_keys)} params")
        if result.missing_keys:
            print(f"  Missing: {len(result.missing_keys)} keys")
        if result.unexpected_keys:
            print(f"  Unexpected: {len(result.unexpected_keys)} keys")

    model.to(device)
    model.eval()

    return model


def _try_remap_state_dict(
    loaded: dict, model_state: dict
) -> dict | None:
    """
    Attempt to remap loaded state dict keys to match the model.

    Handles DomainBed nn.Sequential prefixes (0.model.*, 1.fc.*)
    and FeaturizerClassifierNetwork prefixes (featurizer.model.*, classifier.fc.*).
    """
    model_keys = set(model_state.keys())

    # Try known prefix patterns
    prefix_patterns = [
        # nn.Sequential format from DomainBed algorithms
        ('0.model.', '1.fc.'),
        # FeaturizerClassifierNetwork format
        ('featurizer.model.', 'classifier.fc.'),
    ]

    # Detect classifier head key names in the model
    head_weight_key = None
    head_bias_key = None
    for k in model_keys:
        if any(s in k for s in ['classifier', 'head', 'fc']):
            if 'weight' in k:
                head_weight_key = k
            elif 'bias' in k:
                head_bias_key = k

    for feat_prefix, cls_prefix in prefix_patterns:
        # Check if loaded keys match this pattern
        has_pattern = any(k.startswith(feat_prefix) for k in loaded)
        if not has_pattern:
            continue

        remapped = {}
        for key, value in loaded.items():
            if key.startswith(feat_prefix):
                new_key = key[len(feat_prefix):]
                remapped[new_key] = value
            elif key == f"{cls_prefix}weight" and head_weight_key:
                remapped[head_weight_key] = value
            elif key == f"{cls_prefix}bias" and head_bias_key:
                remapped[head_bias_key] = value

        new_overlap = model_keys & set(remapped.keys())
        if len(new_overlap) > len(model_keys) * 0.5:
            return remapped

    return None


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    num_classes: int,
    task_type: str,
    split: str,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset split cleanly in memory.
    """
    model.eval()
    total_batches = len(dataloader)
    
    # Progress bar setup
    pbar = tqdm(
        total=total_batches,
        desc=f"Evaluating {split}",
        ncols=80,
        disable=not accelerator.is_local_main_process
    )
    
    # Pure in-memory metrics tracking lists
    y_true_list = []
    y_pred_list = []
    prediction_fn = nn.Sigmoid() if task_type == "multi-label" else nn.Softmax(dim=1)
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            # Forward pass
            outputs = model(x)
            
            # Gather predictions across all distributed processes
            gathered_outputs = accelerator.gather_for_metrics(outputs)
            gathered_y = accelerator.gather_for_metrics(y)
            
            # Post-process predictions
            preds = prediction_fn(gathered_outputs)
            
            # Handle potential NaNs safely
            if torch.isnan(preds).any():
                preds = torch.nan_to_num(preds, nan=1.0/num_classes)

            # Keep purely as CPU tensors (extremely memory efficient compared to Python lists/JSON strings)
            y_true_list.append(gathered_y.cpu())
            y_pred_list.append(preds.cpu())
            
            pbar.update(1)
    
    pbar.close()
    
    # Concatenate the collected batches and convert directly to NumPy arrays
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    
    # Slice arrays to match the exact dataset size, trimming distributed sampler padding
    original_dataset_size = len(dataloader.dataset)
    y_true = y_true[:original_dataset_size]
    y_pred = y_pred[:original_dataset_size]
    
    # Calculate and return final metrics
    metrics = compute_metrics(y_true, y_pred, num_classes, task_type)
    return metrics
# =============================================================================
# JSON Handling
# =============================================================================
def load_or_create_metrics_json(metrics_file: Path) -> dict:
    """Load existing metrics JSON or create new structure."""
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {"metrics": {}}


def update_metrics(
    metrics_data: dict,
    dataset: str,
    classifier: str,
    method: str,
    seed: int,
    split_metrics: Dict[str, Dict[str, float]]
) -> dict:
    """
    Update metrics data with evaluation results.
    
    Args:
        metrics_data: Existing metrics dictionary
        dataset: Dataset name
        classifier: Classifier name
        method: Method/augmentation used
        seed: Training seed
        split_metrics: Dictionary of split -> metrics
        
    Returns:
        Updated metrics dictionary
    """
    if "metrics" not in metrics_data:
        metrics_data["metrics"] = {}
    
    # Create nested structure: dataset -> classifier -> method -> seed -> split
    if dataset not in metrics_data["metrics"]:
        metrics_data["metrics"][dataset] = {}
    
    if classifier not in metrics_data["metrics"][dataset]:
        metrics_data["metrics"][dataset][classifier] = {}
    
    if method not in metrics_data["metrics"][dataset][classifier]:
        metrics_data["metrics"][dataset][classifier][method] = {}
    
    seed_str = str(seed)
    metrics_data["metrics"][dataset][classifier][method][seed_str] = split_metrics
    
    return metrics_data


# =============================================================================
# Main Evaluation Function
# =============================================================================
def evaluate_classifier(
    dataset: str,
    data_path: str,
    classifier: str,
    method: str,
    seed: int,
    weights_path: str,
    output_path: str,
    input_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 4,
    splits: Optional[List[str]] = None,
) -> None:
    """
    Evaluate a trained classifier on all specified splits.
    
    Args:
        dataset: Dataset name
        data_path: Path to dataset
        classifier: Classifier architecture name
        method: Training method/augmentation
        seed: Training seed
        weights_path: Path to model weights
        output_path: Path for output metrics JSON
        input_size: Input image size
        batch_size: Evaluation batch size
        num_workers: Number of dataloader workers
        splits: List of splits to evaluate (default: all available)
    """
    # Initialize accelerator
    accelerator = Accelerator()
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Classifier Evaluation")
    accelerator.print(f"{'='*70}")
    accelerator.print(f"Dataset: {dataset}")
    accelerator.print(f"Classifier: {classifier}")
    accelerator.print(f"Method: {method}")
    accelerator.print(f"Seed: {seed}")
    accelerator.print(f"Weights: {weights_path}")
    
    # Get dataset info
    num_classes = NUM_CLASSES[dataset]
    task_type = TASK_TYPE[dataset]
    available_splits = ["val", "val@test_r", "test_r"] 
    
    # Determine splits to evaluate
    if splits is None:
        splits = available_splits
    else:
        splits = [s for s in splits if s in available_splits]
    
    accelerator.print(f"Task type: {task_type}")
    accelerator.print(f"Num classes: {num_classes}")
    accelerator.print(f"Splits to evaluate: {splits}")
    if accelerator.is_main_process:
        output_dir = Path(output_path) / "classifier_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
    # Check if weights exist
    if weights_path != "pretrained" and not Path(weights_path).exists():
        accelerator.print(f"\n✗ Weights not found: {weights_path}")
        return
    
    # Set random seed
    g = random_seed(seed_value=seed)
    
    # Load model
    accelerator.print(f"\nLoading model...")
    if "@" in classifier:
        classifier_name = classifier.split("@")[0]
    else:
        classifier_name = classifier
        
    model = load_classifier(
        weights_path=weights_path,
        classifier=classifier_name,
        num_classes=num_classes,
        device=accelerator.device
    )
    
    # Evaluate on each split
    split_metrics = {}
    
    for split in splits:

        
        accelerator.print(f"\nEvaluating on {split} split...")

        if "@" in split or split in ["test_r", "test_abl", "test_r_c26"]:
            accelerator.print(f"  Applying class subset masking for split: {split}")
            active_model = MaskedClassifier(model, split=split)
            split_num_classes = int(active_model.mask.sum().item())
        else:
            active_model = model
            split_num_classes = num_classes

        eval_model = accelerator.prepare(active_model)

        # Prepare dataloader
        dataloader = prepare_dataloader(
            dataset=dataset,
            data_path=data_path,
            split=split,
            input_size=input_size,
            batch_size=batch_size,
            num_workers=num_workers,
            g=g,
            classifier=classifier_name
        )
        dataloader = accelerator.prepare(dataloader)
        
        # Evaluate
        metrics = evaluate_model(
            model=eval_model,
            dataloader=dataloader,
            accelerator=accelerator,
            num_classes=split_num_classes,
            task_type=task_type,
            split=split,
        )
        
        split_metrics[split] = metrics
        
        # Print results
        accelerator.print(f"  Accuracy: {metrics['accuracy']:.4f}")
        accelerator.print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        accelerator.print(f"  AUC: {metrics['auc']:.4f}")
        accelerator.print(f"  ECE: {metrics['ece']:.4f}")
    
    # Save metrics (only on main process)
    if accelerator.is_main_process:
        metrics_file = output_dir / "metrics.json"
        
        metrics_data = load_or_create_metrics_json(metrics_file)
        metrics_data = update_metrics(
            metrics_data=metrics_data,
            dataset=dataset,
            classifier=classifier,
            method=method,
            seed=seed,
            split_metrics=split_metrics
        )
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        accelerator.print(f"\n✓ Metrics saved to: {metrics_file}")
    
    accelerator.print(f"\n{'='*70}")
    accelerator.print(f"Evaluation Complete")
    accelerator.print(f"{'='*70}")


# =============================================================================
# Main
# =============================================================================
def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained classifiers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--classifier', type=str, required=True,
                       help='Classifier architecture (from timm)')
    parser.add_argument('--method', type=str, required=True,
                       help='Training method/augmentation used')
    parser.add_argument('--seed', type=int, required=True,
                       help='Training seed')
    parser.add_argument('--weights_path', type=str, required=True,
                       help='Path to model weights')
    
    # Optional arguments
    parser.add_argument('--output_path', type=str, default='./results',
                       help='Path for output metrics')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                       help='Splits to evaluate (default: all available)')
    
    # RetriStyle-TTA arguments
    parser.add_argument('--retrieval_mode', type=str, default=None,
                       choices=['random', 'balanced_random', 'ssim', 'mi', 'dino'],
                       help='RetriStyle retrieval mode (None = no TTA)')
    parser.add_argument('--metric_type', type=str, default='ssim',
                       choices=['ssim', 'mi'],
                       help='Structure metric for metric-based retrieval')
    parser.add_argument('--use_kv_cache', action='store_true',
                       help='Use pre-computed KV caches for diffusion')
    parser.add_argument('--offline_db', type=str, default=None,
                       help='Path to pre-computed retrieval database (.pt)')
    parser.add_argument('--n_refs', type=int, default=5,
                       help='Number of style references for RetriStyle TTA')
    
    args = parser.parse_args()
    
    evaluate_classifier(
        dataset=args.dataset,
        data_path=args.data_path,
        classifier=args.classifier,
        method=args.method,
        seed=args.seed,
        weights_path=args.weights_path,
        output_path=args.output_path,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
