"""
xAILab Bamberg
University of Bamberg

@description:
Classification model training script with support for multiple data augmentation strategies
including colorist color transfer augmentation.
"""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import os
import sys
import time
import timm
import psutil
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from torch import Generator
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2, AutoAugmentPolicy
from accelerate import Accelerator
from accelerate.utils import tqdm
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler, create_scheduler_v2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from sklearn.linear_model import LogisticRegression

torch.cuda.empty_cache()

#from medmnistc.augmentation import AugMedMNISTC
#from medmnistc.corruptions.registry import CORRUPTIONS_DS

# Import project modules
from code.experiments.data import create_dataset, NORMALIZATION_MEAN, NORMALIZATION_STD, NUM_CLASSES, TASK_TYPE
from code.experiments.utils.reproducibility import random_seed, worker_seed
from code.experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from code.experiments.utils.training import calculate_passed_time, get_wandb_run_id, get_best_val_loss, get_epochs_no_improve, save_latest_checkpoint, rename_latest_to_final, save_model, get_resume_epoch
from code.experiments.classifier_evaluation import load_classifier
from code.config.constants import PRETRAINED_CLASSIFIERS
from code.config.paths import PROJECT_ROOT

def build_augmentation_transforms(
    augmentations: List[str],
    input_size: int,
    dataset: str,
    exclude_post_augmentations: bool = False,
    color_transfer_params: Optional[Dict[str, Any]] = None
) -> List:
    """
    Build augmentation transform list from augmentation names.
    
    When using batch-level color transfer, this builds transforms BEFORE color transfer.
    Post-augmentations (random_erasing) are applied AFTER color transfer in the training loop.
    
    Args:
        augmentations: List of augmentation names to apply
        input_size: Target input size for the model
        dataset: Dataset name (for targeted augmentation)
        exclude_post_augmentations: If True, excludes random_erasing (applied after color transfer)
        color_transfer_params: Dictionary with 'color_space', 'method', 'p' for color transfer (deprecated for batch-level)
        
    Returns:
        List of transforms (not yet composed)
    """
    transforms = []
    
    # Add each augmentation to the pipeline
    for aug in augmentations:
        if aug == 'none':
            # No augmentation
            continue
            
        elif aug == 'gray_scale':
            transforms.append(v2.Grayscale(num_output_channels=3))
        
        elif aug == 'color_jitter':
            transforms.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
        
        elif aug == 'auto_augment':
            transforms.append(v2.AutoAugment())
        
        elif aug == 'rand_augment':
            transforms.append(v2.RandAugment())
        
        elif aug == 'trivial_augment':
            transforms.append(v2.TrivialAugmentWide())
        
        elif aug == 'aug_mix':
            transforms.append(v2.AugMix())
        
        elif aug == 'random_resized_crop':
            transforms.append(v2.RandomResizedCrop(size=(input_size, input_size)))
        
        elif aug == 'random_flip':
            transforms.append(v2.RandomHorizontalFlip())
            transforms.append(v2.RandomVerticalFlip())
        
        elif aug == 'random_erasing':
            # Skip if exclude_post_augmentations (applied after color transfer)
            if not exclude_post_augmentations:
                transforms.append(v2.RandomErasing())
                
        elif aug == 'targeted_augment':
            if "mnist" in dataset:
                transforms.append(AugMedMNISTC(CORRUPTIONS_DS[dataset]))
            elif dataset in ["camelyon17wilds", "epistr"]:
                transforms.append(AugMedMNISTC(CORRUPTIONS_DS["pathmnist"]))
            elif dataset in ["fitzpatrick17k", "ddi"]:
                transforms.append(AugMedMNISTC(CORRUPTIONS_DS["dermamnist"]))
            else:
                raise ValueError(f"Targeted augmentation not available for dataset: {dataset}")
        
        elif aug == 'color_transfer':
            # Skip - color transfer is applied at batch level for efficiency
            # Old per-sample approach (commented out for future use):
            # transforms.append(
            #     RandomColorTransfer(
            #         color_space=color_transfer_params.get('color_space', 'rgb'),
            #         method=color_transfer_params.get('method', 'mean_std'),
            #         p=color_transfer_params.get('p', 0.5)
            #     )
            # )
            pass
        
        else:
            raise ValueError(f"Unknown augmentation: {aug}.")
    
    return transforms


def prepare_dataloaders(
    dataset: str,
    data_path: str,
    input_size: int,
    batch_size: int,
    num_workers: int,
    augmentations: List[str],
    split: Optional[str],
    color_transfer_params: Optional[Dict[str, Any]],
    g: Generator,
    classifier: Optional[str] = None,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation dataloaders with augmentation.
    
    Args:
        dataset: Dataset name
        data_path: Path to dataset
        input_size: Input size for model
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        augmentations: List of augmentation names
        color_transfer_params: Parameters for color transfer (if used)
        g: Random generator for reproducibility
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    if classifier == 'dinov2_vitb14' and input_size % 14 != 0:
        raise ValueError(f"DINOv2 requires input_size divisible by 14, got {input_size}")
    
    stats_name = dataset if not "ViT" in classifier else "ViT"
    mean=NORMALIZATION_MEAN[stats_name]
    std=NORMALIZATION_STD[stats_name]

    extract = kwargs.get("extraction", False)

    # Check if using color transfer (applied at batch level)
    use_color_transfer = 'color_transfer' in augmentations
    
    # Build augmentation pipeline (excluding post-augmentations if using color transfer)
    train_augmentations = build_augmentation_transforms(
        augmentations=augmentations,
        input_size=input_size,
        dataset=dataset,
        exclude_post_augmentations=use_color_transfer,  # Exclude random_erasing if using color transfer
        color_transfer_params=color_transfer_params
    )
    
    # Training transform
    # When using color transfer: normalization is applied AFTER color transfer at batch level
    # Colorist transforms expect [0, 1] range, not normalized images
    if use_color_transfer:
        if 'targeted_augment' in augmentations:
            # Targeted augmentations expect PIL images, so convert to tensors after applying the augmentation
            train_transform = v2.Compose([
                ResizeWhileRetainAspectRatio(size=input_size),
                *train_augmentations,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                # No normalization here - applied after color transfer
            ])
        
        else:
            train_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                ResizeWhileRetainAspectRatio(size=input_size),
                *train_augmentations,
                # No normalization here - applied after color transfer
            ])
    else:
        if 'targeted_augment' in augmentations:
            # Targeted augmentations expect PIL images, so convert to tensors after applying the augmentation
            train_transform = v2.Compose([
                ResizeWhileRetainAspectRatio(size=input_size),
                *train_augmentations,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std)
            ])
        else:
            train_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                #ResizeWhileRetainAspectRatio(size=input_size),
                *train_augmentations,
                v2.Normalize(mean, std)
            ])
    
    # Validation transform
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=input_size),
        v2.Normalize(mean, std)
    ])
    
    # Load datasets
    # Color transfer is applied at batch level, so always use transform
    # Old per-sample approach (commented out for future use):
    # if use_color_transfer:
    #     train_set = create_dataset(..., transform=None, ...)
    #     train_set = ColorTransferDataset(train_set, transform=train_transform)

    if 1==2: #extract and classifier in PRETRAINED_CLASSIFIERS:
        classifier = classifier.replace(".", "_") if "." in classifier else classifier
        cache = Path(f"./data/embeddings/{classifier}/{dataset}")
        if cache.exists():
            print(f"Loading cached features from {cache}")
            train_data = torch.load(cache / 'train.pt', weights_only=True)
            train_set = TensorDataset(train_data["features"], train_data["labels"])
            
            val_data = torch.load(cache / 'val.pt', weights_only=True)
            val_set = TensorDataset(val_data["features"], val_data["labels"])
        else:
            raise FileNotFoundError("Could not find cache dict")
    else:
        train_set = create_dataset(
            dataset_name=dataset,
            data_path=data_path,
            split="train" if not split else f"train@{split}",
            transform=train_transform,
            **kwargs
        )
        
        val_set = create_dataset(
            dataset_name=dataset,
            data_path=data_path,
            split="val" if not split else f"val@{split}",
            transform=val_transform,
            **kwargs
        )
        
    # Create dataloaders

    targets = [label for _, label in train_set.samples] if hasattr(train_set, 'samples') else [train_set[i][1] for i in range(len(train_set))]
    class_sample_count = np.bincount(targets)
    weights_per_class = 1.0 / class_sample_count
    sample_weights = [weights_per_class[t] for t in targets]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
        sampler = sampler,
        generator=g,
    )
    
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
        generator=g,
    )
    
    return train_loader, val_loader, dataset


def create_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    num_epochs: int
) -> Tuple[Optimizer, CosineLRScheduler, int]:
    """
    Create optimizer and learning rate scheduler.
    
    Uses Adam optimizer and cosine annealing scheduler with default parameters.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (optimizer, lr_scheduler, updated_num_epochs)
    """
    """# Create Adam optimizer
    optimizer = create_optimizer_v2(
        model,
        opt='adam',
        lr=lr,
    )
    
    # Create cosine annealing scheduler
    
    lr_scheduler, _ = create_scheduler_v2(
        optimizer=optimizer,
        sched='cosine',
        num_epochs=num_epochs,
        warmup_epochs=5,
        min_lr=1e-6,
    )"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        raise ValueError(
            f"No trainable parameters found in the model. "
            f"Check if layers were accidentally frozen completely."
        )

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=lr,
        momentum=0.9,
        weight_decay=0
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs  # Matches your dynamic training bounds
    )
    
    return optimizer, lr_scheduler


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
        task_type: Type of task ("multi-class" or "multi-label")
        
    Returns:
        Dictionary with accuracy, balanced_accuracy, and auc
    """
    if task_type == "multi-label":
        # Multi-label classification
        y_pred_labels = (y_pred > 0.5).astype(int)
        # For multi-label, compute sample-wise accuracy
        accuracy = (y_pred_labels == y_true).all(axis=1).mean()
        # Balanced accuracy not applicable for multi-label
        balanced_acc = accuracy
        # AUC for multi-label (sample-averaged)
        #auc = roc_auc_score(y_true, y_pred, average='samples')
    elif num_classes == 2:
        # Binary classification
        y_true_squeezed = y_true.squeeze()
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)
        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        #auc = roc_auc_score(y_true_squeezed, y_pred[:, -1])
    else:
        # Multi-class classification
        y_true_squeezed = y_true.astype(int).flatten()
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)
        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        #auc = roc_auc_score(y_true_squeezed, y_pred, multi_class="ovr")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_criterion: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: CosineLRScheduler,
    accelerator: Accelerator,
    epoch: int,
    num_epochs: int,
    num_classes: int,
    task_type: str,
    color_transfer_fn = None,
    color_transfer_prob: float = 0.5,
    post_color_transfer_transform: Optional[v2.Compose] = None,
    normalization_mean: Optional[List[float]] = None,
    normalization_std: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training dataloader
        loss_criterion: Loss function
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        accelerator: Accelerator for distributed training
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        num_classes: Number of classes
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    # Progress bar
    pbar = tqdm(
        total=len(train_loader),
        bar_format="{l_bar}{bar}",
        ncols=80,
        initial=0,
        position=0,
        leave=False,
    )
    pbar.set_description(f"Train [{epoch + 1}/{num_epochs}]")
    
    # Metrics tracking
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    prediction_fn = nn.Sigmoid() if task_type == "multi-label" else nn.Softmax(dim=1)
    
    # Number of updates for lr scheduler
    num_updates = epoch * len(train_loader)
    
    # Training loop with gradient accumulation
    for batch_idx, (x, y) in enumerate(train_loader):
        # Apply color transfer at batch level if configured
        if color_transfer_fn is not None:
            # Determine how many samples to augment
            num_samples_to_augment = max(1, int(color_transfer_prob * len(x)))
            
            # Select random samples to augment
            idx_samples_to_augment = torch.randperm(len(x), device=x.device)[:num_samples_to_augment]
            x_content = x[idx_samples_to_augment].clone()
            
            # Create reference batch by rolling (each sample gets next sample as reference)
            x_reference = x_content.roll(shifts=-1, dims=0)
            
            # Apply color transfer (colorist functions expect [0,1] range)
            x_transferred = color_transfer_fn(source=x_content, reference=x_reference)

            # Map transferred images to the GPU
            x_transferred = x_transferred.to(accelerator.device)

            # Check for NaNs/Infs (Stability fix)
            if torch.isnan(x_transferred).any() or torch.isinf(x_transferred).any():
                accelerator.print(f"Warning: NaNs/Infs detected in color transfer output at epoch {epoch+1}, batch {batch_idx}. Skipping augmentation.")
                x_transferred = x_content.to(accelerator.device) # Fallback
            
            # Ensure strictly [0, 1] range before normalization
            x_transferred = x_transferred.clamp(0.0, 1.0)
            
            # Replace augmented samples
            x[idx_samples_to_augment] = x_transferred
            
            # Apply post-color-transfer augmentations (e.g., RandomErasing)
            if post_color_transfer_transform is not None:
                x = torch.stack([post_color_transfer_transform(img) for img in x])
            
            # Apply normalization AFTER color transfer (colorist expects [0,1] range)
            if normalization_mean is not None and normalization_std is not None:
                normalize = v2.Normalize(mean=normalization_mean, std=normalization_std)
                x = torch.stack([normalize(img) for img in x])
        
        with accelerator.accumulate(model):
            # Forward pass
            outputs = model(x)
            
            # Compute loss (handle multi-label vs multi-class)
            if task_type == "multi-label":
                y_float = y.to(torch.float32)
                loss = loss_criterion(outputs, y_float)
            else:
                y_squeezed = y.squeeze().long()
                loss = loss_criterion(outputs, y_squeezed)
            
            # Check for NaNs in loss
            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"Warning: NaN/Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping optimization step.")
                optimizer.zero_grad() 
                continue # Skip the rest of the loop for this batch

            # Backward pass
            accelerator.backward(loss)

            # Gradient clipping to prevent exploding gradients
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)

            optimizer.step()
            if hasattr(lr_scheduler, 'step_update'):
                lr_scheduler.step_update(num_updates=num_updates)
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item()
        y_true_list.append(y.cpu())

        global_step = epoch * len(train_loader) + batch_idx
        if accelerator.is_main_process:
            accelerator.log({
                "train/step_loss": loss.item(),
                "train/lr": optimizer.param_groups[0]['lr']
            }, step=global_step)
        
        # Handle potential NaNs in outputs for metrics (though less likely if loss is valid)
        preds = prediction_fn(outputs).detach().cpu()
        if torch.isnan(preds).any():
             # Replace NaNs with uniform distribution or zeros to prevent metric crash
             preds = torch.nan_to_num(preds, nan=1.0/num_classes)
        y_pred_list.append(preds)
        
        # Update progress bar
        pbar.update(1)
        num_updates += 1
    
    # Update scheduler at epoch end
    if hasattr(lr_scheduler, 'step_update'):
        lr_scheduler.step(epoch + 1 )
    else:
        lr_scheduler.step()
    # Compute metrics
    avg_loss = total_loss / len(train_loader)
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()
    metrics = compute_metrics(y_true, y_pred, num_classes, task_type)
    metrics['loss'] = avg_loss
    
    return metrics


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_criterion: nn.Module,
    accelerator: Accelerator,
    epoch: int,
    num_epochs: int,
    num_classes: int,
    task_type: str
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation dataloader
        loss_criterion: Loss function
        accelerator: Accelerator for distributed training
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        num_classes: Number of classes
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    # Progress bar
    pbar = tqdm(
        total=len(val_loader),
        bar_format="{l_bar}{bar}",
        ncols=80,
        initial=0,
        position=0,
        leave=False,
    )
    pbar.set_description(f"Val [{epoch + 1}/{num_epochs}]")
    
    # Metrics tracking
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    prediction_fn = nn.Sigmoid() if task_type == "multi-label" else nn.Softmax(dim=1)
    
    # Validation loop
    with torch.no_grad():
        for x, y in val_loader:
            # Forward pass
            outputs = model(x)
            
            # Compute loss (handle multi-label vs multi-class)
            if task_type == "multi-label":
                y_float = y.to(torch.float32)
                loss = loss_criterion(outputs, y_float)
            else:
                y_squeezed = y.squeeze().long()
                loss = loss_criterion(outputs, y_squeezed)
            
            # Gather metrics for consistent validation across processes
            gathered_outputs = accelerator.gather_for_metrics(outputs)
            gathered_y = accelerator.gather_for_metrics(y)
            
            # Reduce loss to get global average
            reduced_loss = accelerator.reduce(loss, reduction="mean")
            
            # Track metrics
            total_loss += reduced_loss.item()
            y_true_list.append(gathered_y.cpu())
            
            # Handle potential NaNs in outputs for metrics
            preds = prediction_fn(gathered_outputs).cpu()
            if torch.isnan(preds).any():
                 # Replace NaNs with uniform distribution or zeros to prevent metric crash
                 preds = torch.nan_to_num(preds, nan=1.0/num_classes)
            y_pred_list.append(preds)
            
            # Update progress bar
            pbar.update(1)
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    y_true = torch.cat(y_true_list).numpy()
    y_pred = torch.cat(y_pred_list).numpy()

    
    metrics = compute_metrics(y_true, y_pred, num_classes, task_type)
    metrics['loss'] = avg_loss
    
    return metrics


def train(
    dataset: str,
    data_path: str,
    output_path: str,
    checkpoint_path: str,
    classifier: str,
    input_size: int,
    augmentations: List[str],
    color_transfer_space: Optional[str],
    color_transfer_method: str,
    color_transfer_prob: float,
    epochs: int,
    early_stopping: int,
    batch_size: int,
    lr: float,
    seed: int,
    use_cuda: bool,
    num_workers: int,
    split: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    save_checkpoint_every: int = 10,
    gradient_accumulation_steps: int = 1,
    train_mode: str = "linear_probe",
    **kwargs
) -> None:
    """
    Main training function.
    
    Args:
        dataset: Dataset name
        data_path: Path to dataset
        output_path: Path to save final trained models
        checkpoint_path: Path to save checkpoints for resuming training
        classifier: Classifier model name (from timm)
        input_size: Input image size
        augmentations: List of augmentation names
        color_transfer_space: Color space for color transfer (if used)
        color_transfer_method: Method for color transfer
        color_transfer_prob: Probability of applying color transfer
        epochs: Number of training epochs
        early_stopping: Number of epochs without improvement before stopping
        batch_size: Training batch size
        lr: Learning rate
        seed: Random seed
        use_cuda: Whether to use CUDA
        num_workers: Number of dataloader workers
        use_wandb: Whether to use Weights and Biases for logging
        wandb_project: WandB project name
        wandb_entity: WandB entity name
        wandb_path: Directory for WandB logs (default: wandb/ in current directory)
        resume_from_checkpoint: Whether to resume from latest checkpoint
        save_checkpoint_every: Save checkpoint every N epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating
        **kwargs: Additional arguments for dataset creation
    """

    # Create output and checkpoint directories
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create checkpoint save directory and final model weights filename
    aug_str = '-'.join(sorted(augmentations))
    if 'color_transfer' in augmentations:
        prob_percent = int(color_transfer_prob * 100)
        aug_str = f"{aug_str}-{color_transfer_space}-direct-{color_transfer_method}-{prob_percent}percent"
    
    classifier_name = classifier.replace(".", "_") if "." in classifier else classifier
    split_name = f"-{split}-" if split else ""
    run_name = f"{dataset}{split_name}-{classifier_name}-{aug_str}-seed{seed}"

    latest_checkpoint_path = checkpoint_path / f"{run_name}_latest"
    final_model_path = output_path / f"{run_name}.pth"
        
    # Prepare color transfer parameters if needed
    color_transfer_params = None
    if 'color_transfer' in augmentations:
        if color_transfer_space is None:
            raise ValueError("color_transfer_space must be specified when using color_transfer augmentation")
        
        color_transfer_params = {
            'color_space': color_transfer_space,
            'method': color_transfer_method,
            'p': color_transfer_prob
        }

    # Initialize accelerator with gradient accumulation, mixed precision, and logging
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=PROJECT_ROOT / "models" / "runs"
    )
        
   
    accelerator.init_trackers(
        project_name=run_name,
        config={
            'dataset': dataset,
            'classifier': classifier,
            'augmentations': augmentations,
            'color_transfer_space': color_transfer_space,
            'color_transfer_method': color_transfer_method,
            'color_transfer_prob': color_transfer_prob,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'seed': seed,
        }
    )
    
    # Set random seed
    accelerator.print(f"Setting random seed: {seed}")
    g = random_seed(seed_value=seed, use_cuda=use_cuda)

    # Scale batch size by number of processes to ensure global batch size matches argument
    per_device_batch_size = batch_size // accelerator.num_processes
    accelerator.print(f"Global batch size: {batch_size}, Per-device batch size: {per_device_batch_size} (across {accelerator.num_processes} processes)")
    
    # Prepare dataloaders
    accelerator.print(f"Loading dataset: {dataset}")

    train_loader, val_loader, dataset_name = prepare_dataloaders(
        dataset=dataset,
        data_path=data_path,
        input_size=input_size,
        batch_size=per_device_batch_size,
        num_workers=num_workers,
        augmentations=augmentations,
        color_transfer_params=color_transfer_params,
        g=g,
        classifier=classifier,
        split=split, 
        extraction=True,
        **kwargs
    )
    
    # Load color transfer function if needed (for batch-level processing)
    color_transfer_fn = None
    use_color_transfer = 'color_transfer' in augmentations
    post_color_transfer_transform = None
    
    if use_color_transfer:
        import warnings
        warnings.warn(
            "Legacy colorist color_transfer augmentation has been removed. "
            "Use style_tta for test-time adaptation instead. "
            "Continuing without color transfer augmentation.",
            DeprecationWarning,
            stacklevel=2,
        )
        use_color_transfer = False
        accelerator.print("WARNING: color_transfer augmentation disabled (colorist removed). Use style_tta TTA at eval time.")
        
        # Build post-color-transfer augmentations (e.g., RandomErasing)
        post_transforms = []
        if 'random_erasing' in augmentations:
            post_transforms.append(v2.RandomErasing())
        
        if post_transforms:
            post_color_transfer_transform = v2.Compose(post_transforms)
        
        accelerator.print(f"✓ Color transfer function loaded")
    
    # Create model
    accelerator.print(f"Creating model: {classifier}")
    num_classes = NUM_CLASSES[dataset]

    # 1. Base Model loading (Ensuring pre-trained parameters are ALWAYS pulled)
    if classifier in PRETRAINED_CLASSIFIERS:
        model = load_classifier(
            classifier=classifier,
            num_classes=num_classes,
            weights_path="linear_probe", # Base backbone weights initialization
            device='cuda'
        )
    else:
        import timm
        model = timm.create_model(classifier, pretrained=True, num_classes=num_classes)

    # 2. Strategy Assignment (Unfreezing Parameters dynamically)
    if train_mode == "linear_probe":
        model.requires_grad_(False)
        # Handle structural variance across timm architectural hooks
        if hasattr(model, 'head'):
            model.head.requires_grad_(True)
        elif hasattr(model, 'fc'):
            model.fc.requires_grad_(True)
        elif hasattr(model, 'classifier'):
            model.classifier.requires_grad_(True)
            
    elif train_mode == "finetune":
        model.requires_grad_(True)

    # Create optimizer and scheduler
    accelerator.print("Creating optimizer and scheduler")
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=lr,
        num_epochs=epochs
    )
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    accelerator.print(f"Trainable params: {trainable}")
    accelerator.print(f"Total trainable: {sum(n for _, n in trainable)}")
    # Create loss function
    task_type = TASK_TYPE[dataset]
    if task_type == "multi-label":
        loss_criterion = nn.BCEWithLogitsLoss()
    else:
        loss_criterion = nn.CrossEntropyLoss()
    
    # Prepare for distributed training
    model, loss_criterion, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model, loss_criterion, optimizer, lr_scheduler, train_loader, val_loader
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_val_loss = np.inf
    epochs_no_improve = 0
    
    if resume_from_checkpoint:
        if latest_checkpoint_path.exists() and any(latest_checkpoint_path.iterdir()):
            try:
                accelerator.wait_for_everyone()
                accelerator.load_state(str(latest_checkpoint_path))
                accelerator.print(f"\nResuming from checkpoint: {latest_checkpoint_path}")
                
                # Get resume epoch from metadata
                start_epoch = get_resume_epoch(latest_checkpoint_path)
                
                # Restore best_val_loss and epochs_no_improve for early stopping
                best_val_loss = get_best_val_loss(latest_checkpoint_path)
                epochs_no_improve = get_epochs_no_improve(latest_checkpoint_path)
                
                accelerator.print(f"Resuming from epoch {start_epoch + 1}")
                accelerator.print(f"Best val loss so far: {best_val_loss:.4f}")
                accelerator.print(f"Epochs without improvement: {epochs_no_improve}")
            except Exception as e:
                accelerator.print(f"Error loading checkpoint: {e}")
                accelerator.print("Starting training from scratch.")
                start_epoch = 0
                best_val_loss = np.inf
                epochs_no_improve = 0
        else:
            accelerator.print("No checkpoint found. Starting training from scratch.")
    else:
        accelerator.print("Starting training from scratch.")
    
    # Training loop
    accelerator.print("Starting training")
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_criterion=loss_criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            epoch=epoch,
            num_epochs=epochs,
            num_classes=num_classes,
            task_type=task_type,
            color_transfer_fn=color_transfer_fn,
            color_transfer_prob=color_transfer_prob,
            post_color_transfer_transform=post_color_transfer_transform,
            normalization_mean=NORMALIZATION_MEAN[dataset_name] if use_color_transfer else None,
            normalization_std=NORMALIZATION_STD[dataset_name] if use_color_transfer else None
        )
        
        # Validate
        val_metrics = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            loss_criterion=loss_criterion,
            accelerator=accelerator,
            epoch=epoch,
            num_epochs=epochs,
            num_classes=num_classes,
            task_type=task_type
        )
        
        # Log metrics
        logs = {
                "epoch": epoch + 1,
                "train/loss": train_metrics['loss'],
                "train/accuracy": train_metrics['accuracy'],
                "train/balanced_accuracy": train_metrics['balanced_accuracy'],
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/balanced_accuracy": val_metrics['balanced_accuracy'],
                "lr": optimizer.param_groups[0]['lr'],
            }

        #Log via Accelerate (wandb/tensorboard)
        accelerator.log(logs, step=epoch + 1)

        if accelerator.is_main_process:
            log_file = output_path / f"{run_name}_metrics.csv"
            file_exists = log_file.exists()
            with open(log_file, mode='a') as f:
                if not file_exists:
                    f.write(",".join(logs.keys()) + "\n")
                f.write(",".join(map(str, logs.values())) + "\n")
        
        # Print metrics
        accelerator.print(f"\nEpoch [{epoch + 1}/{epochs}]")
        accelerator.print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                         f"Bal Acc: {train_metrics['balanced_accuracy']:.4f})")#, AUC: {train_metrics['auc']:.4f}")
        accelerator.print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                         f"Bal Acc: {val_metrics['balanced_accuracy']:.4f}")#, AUC: {val_metrics['auc']:.4f}")
        
        # Save current best model
        if val_metrics['loss'] < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_metrics['loss']
            save_model(accelerator, model, final_model_path)

        else:
            epochs_no_improve += 1
                
        # Save checkpoint periodically
        if (epoch + 1) % save_checkpoint_every == 0:
            accelerator.print(f"Saving checkpoint at epoch {epoch + 1}...")
            
            # Get WandB run ID if available
            checkpoint_metadata = {
                'epoch': epoch + 1,
                'val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }
            
            save_latest_checkpoint(
                accelerator,
                latest_checkpoint_path,
                metadata=checkpoint_metadata
            )
        
        # Epoch time
        hours, minutes, seconds = calculate_passed_time(epoch_start, time.time())
        accelerator.print(f"Epoch time: {hours:02d}:{minutes:02d}:{seconds:05.2f}\n")
        
        # Early stopping
        if epochs_no_improve >= early_stopping:
            accelerator.print(f"Early stopping after {epoch + 1} epochs")
            break
    
    # Save final checkpoint
    accelerator.print("\n" + "="*80)
    accelerator.print("Training Complete!")
    accelerator.print("="*80)
    
    accelerator.print("Saving final checkpoint...")
    
    # Get WandB run ID if available
    final_metadata = {
        'epoch': epochs,
        'val_loss': val_metrics['loss'],
        'best_val_loss': best_val_loss,
        'training_complete': True
    }

    
    save_latest_checkpoint(
        accelerator,
        latest_checkpoint_path,
        metadata=final_metadata
    )
    
    # Rename latest to final (only main process)
    if accelerator.is_main_process:
        rename_latest_to_final(latest_checkpoint_path)
    
    # End training
    accelerator.end_training()
    
    # Total time
    hours, minutes, seconds = calculate_passed_time(start_time, time.time())
    accelerator.print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description='Train classification model with multiple augmentation strategies',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_path', type=str, default=str(PROJECT_ROOT / 'models'), help='Path to save final trained models')
    parser.add_argument('--checkpoint_path', type=str, default=str(PROJECT_ROOT / 'models' / 'checkpoints'), help='Path to save checkpoints for resuming training')
    parser.add_argument('--train_mode', type=str, default='linear_probe', help='Trainigs Mode full finetune or just linear probe')
    # Add this under Dataset configuration in main():
    parser.add_argument('--split', type=str, default=None, 
                        help='Dataset split to train on (e.g., train, train@breasts)')
    # Model configuration
    parser.add_argument('--classifier', type=str, default='resnet50', help='Classifier model name (from timm)')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    
    # Augmentation configuration
    parser.add_argument('--augmentations', type=str, nargs='+', default=['none'],
                       help='List of augmentations to apply. Options: none, gray_scale, color_jitter, '
                            'auto_augment, rand_augment, trivial_augment, aug_mix, random_resized_crop, '
                            'random_horizontal_flip, random_vertical_flip, random_rotation, '
                            'targeted_augment, color_transfer')
    parser.add_argument('--color_transfer_space', type=str, default='lab', help='Color space for color transfer (required if color_transfer in augmentations)')
    parser.add_argument('--color_transfer_method', type=str, default='mean_std', choices=['mean_std', 'histogram', 'ehm'], help='Color transfer method')
    parser.add_argument('--color_transfer_prob', type=float, default=0.5, help='Probability of applying color transfer')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=10, help='Number of epochs without improvement before early stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps * num_gpus)')
    
    # Optimizer configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # System configuration
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # Checkpoint configuration
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume training from latest checkpoint if available')
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Validate arguments
    if 'color_transfer' in args.augmentations and args.color_transfer_space is None:
        parser.error("--color_transfer_space is required when using color_transfer augmentation")
    
    # Start training
    train(
        dataset=args.dataset,
        split=args.split,
        data_path=args.data_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        classifier=args.classifier,
        input_size=args.input_size,
        augmentations=args.augmentations,
        color_transfer_space=args.color_transfer_space,
        color_transfer_method=args.color_transfer_method,
        color_transfer_prob=args.color_transfer_prob,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_cuda=args.use_cuda,
        num_workers=args.num_workers,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_checkpoint_every=args.save_checkpoint_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_mode=args.train_mode, 
    )


if __name__ == "__main__":
    main()
    """
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    for model in ["ViT-B-16", "dinov2_vitb14"]:
        train_lbfgs(
            train_cache=f"./data/feature_cache/{model}/train.pt",
            val_cache=f"./data/feature_cache/{model}/val.pt",
            num_classes=1000,
            device=device,
            num_iter=50,
            output_path="./data/models",
            run_name=f"{model}"
        )
    """
