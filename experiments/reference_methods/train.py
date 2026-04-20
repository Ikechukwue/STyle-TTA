"""
xAILab Bamberg
University of Bamberg

@description:
Classification model training script using reference-based augmentation methods (AdaIN, AdaAttN, etc.)
as data augmentation instead of custom color transfer.
"""

import os
import sys
import time
import timm
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from torch import Generator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import v2, AutoAugmentPolicy
from accelerate import Accelerator
from accelerate.utils import tqdm
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler, create_scheduler_v2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from medmnistc.augmentation import AugMedMNISTC
from medmnistc.corruptions.registry import CORRUPTIONS_DS

# Import project modules
from experiments.data import create_dataset, NORMALIZATION_MEAN, NORMALIZATION_STD, NUM_CLASSES, TASK_TYPE
from experiments.utils.reproducibility import random_seed, worker_seed
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.utils.training import calculate_passed_time, get_wandb_run_id, get_best_val_loss, get_epochs_no_improve, save_latest_checkpoint, rename_latest_to_final, save_model, get_resume_epoch
from experiments.reference_methods.style_transfer_factory import create_color_transfer_method


def build_augmentation_transforms(
    augmentations: List[str],
    input_size: int,
    dataset: str,
    exclude_post_augmentations: bool = False
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
        
        elif aug == 'random_erasing' and not exclude_post_augmentations:
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
    g: Generator,
    color_transfer_native_size: Optional[int] = None,
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
        g: Random generator for reproducibility
        color_transfer_native_size: Native image size for color transfer method (if using color transfer)
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Check if using color transfer (applied at batch level)
    use_color_transfer = 'color_transfer' in augmentations
    
    # Build augmentation pipeline (excluding post-augmentations if using color transfer)
    train_augmentations = build_augmentation_transforms(
        augmentations=augmentations,
        input_size=input_size,
        dataset=dataset,
        exclude_post_augmentations=use_color_transfer  # Exclude random_erasing if using color transfer
    )
    
    # Training transform
    # When using color transfer: normalization is applied AFTER color transfer at batch level
    # Color transfer models are trained on [0, 1] range, not normalized images
    # Use native image size of color transfer method if provided
    if use_color_transfer:
        resize_size = color_transfer_native_size if color_transfer_native_size is not None else input_size
        
        if 'targeted_augment' in augmentations:
            # Targeted augmentations expect PIL images, so convert to tensors after applying the augmentation
            train_transform = v2.Compose([
                ResizeWhileRetainAspectRatio(size=resize_size),
                *train_augmentations,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                # No normalization here - applied after color transfer
            ])
        
        else:
            train_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                ResizeWhileRetainAspectRatio(size=resize_size),
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
                v2.Normalize(mean=NORMALIZATION_MEAN[dataset], std=NORMALIZATION_STD[dataset])
            ])
        else:
            train_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                ResizeWhileRetainAspectRatio(size=input_size),
                *train_augmentations,
                v2.Normalize(mean=NORMALIZATION_MEAN[dataset], std=NORMALIZATION_STD[dataset])
            ])
    
    # Validation transform
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=input_size),
        v2.Normalize(mean=NORMALIZATION_MEAN[dataset], std=NORMALIZATION_STD[dataset])
    ])
    
    # Load datasets
    # Color transfer is applied at batch level, so always use transform
    train_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split="train",
        transform=train_transform,
        **kwargs
    )
    
    val_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split="val",
        transform=val_transform,
        **kwargs
    )
        
    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
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
    
    return train_loader, val_loader


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
    # Create Adam optimizer
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
        auc = roc_auc_score(y_true, y_pred, average='samples')
    elif num_classes == 2:
        # Binary classification
        y_true_squeezed = y_true.squeeze()
        y_pred_labels = (y_pred[:, -1] > 0.5).astype(int)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)
        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        auc = roc_auc_score(y_true_squeezed, y_pred[:, -1])
    else:
        # Multi-class classification
        y_true_squeezed = y_true.squeeze()
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)
        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        auc = roc_auc_score(y_true_squeezed, y_pred, multi_class="ovr")
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'auc': auc
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
    input_size: int,
    color_transfer_transform: Optional[nn.Module] = None,
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
        input_size: Input size for classifier (used to resize after color transfer)
        color_transfer_transform: Optional transform for batch-level color transfer
        color_transfer_prob: Probability of applying color transfer to batch
        post_color_transfer_transform: Optional transforms to apply after color transfer
        normalization_mean: Mean for normalization (applied after color transfer)
        normalization_std: Std for normalization (applied after color transfer)
        
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
        if color_transfer_transform is not None:
            # Determine how many samples to augment
            num_samples_to_augment = max(1, int(color_transfer_prob * len(x)))
            
            # Select random samples to augment
            idx_samples_to_augment = torch.randperm(len(x))[:num_samples_to_augment]
            x_content = x[idx_samples_to_augment].clone()
            
            # Create style batch by rolling (each sample gets next sample as style)
            x_style = x_content.roll(shifts=-1, dims=0)
            
            # Apply color transfer on GPU
            with torch.no_grad():
                x_transferred = color_transfer_transform(x_content, x_style)

            # Check for NaNs/Infs in output
            if torch.isnan(x_transferred).any() or torch.isinf(x_transferred).any():
                accelerator.print(f"Warning: NaNs/Infs detected in color transfer output at epoch {epoch+1}, batch {batch_idx}. Skipping augmentation.")
                x_transferred = x_content
            
            # Ensure strictly [0, 1] range before normalization
            x_transferred = x_transferred.clamp(0.0, 1.0)
            
            # Replace augmented samples
            x[idx_samples_to_augment] = x_transferred
            
            # Resize back to classifier input size (color transfer may use different native size)
            # This ensures the classifier receives images at the expected resolution
            x = v2.Resize(size=(input_size, input_size), antialias=True)(x)
            
            # Apply post-color-transfer augmentations (e.g., RandomErasing)
            if post_color_transfer_transform is not None:
                x = torch.stack([post_color_transfer_transform(img) for img in x])
            
            # Apply normalization AFTER color transfer (models trained on [0,1] range)
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
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()
            lr_scheduler.step_update(num_updates=num_updates)
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item()
        y_true_list.append(y.cpu())
        
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
    lr_scheduler.step(epoch + 1)
    
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
            # This prevents deadlocks where one process saves best model and others don't
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
    color_transfer_method: Optional[str],
    color_transfer_weights: Optional[str],
    color_transfer_prob: float,
    epochs: int,
    early_stopping: int,
    batch_size: int,
    lr: float,
    seed: int,
    use_cuda: bool,
    num_workers: int,
    use_wandb: bool,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_path: Optional[str] = None,
    resume_from_checkpoint: bool = False,
    save_checkpoint_every: int = 10,
    gradient_accumulation_steps: int = 1,
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
        color_transfer_method: Reference augmentation method name ('adain', 'adaattn', etc.)
        color_transfer_weights: Path to pretrained method weights
        color_transfer_prob: Probability of applying reference transform
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
        wandb_path: Directory for WandB logs
        resume_from_checkpoint: Whether to resume from latest checkpoint
        save_checkpoint_every: Save checkpoint every N epochs
        gradient_accumulation_steps: Number of steps to accumulate gradients
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
        aug_str = f"{aug_str}-{color_transfer_method}-{prob_percent}percent"
    run_name = f"{dataset}-{classifier}-{aug_str}-seed{seed}"

    latest_checkpoint_path = checkpoint_path / f"{run_name}_latest"
    final_model_path = output_path / f"{run_name}.pth"
        
    # Set WandB directory if specified
    if wandb_path is not None:
        os.environ['WANDB_DIR'] = wandb_path
        Path(wandb_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize accelerator
    log_with = ["wandb"] if use_wandb else None
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=log_with
    )
        
    # Check if resuming from checkpoint
    wandb_run_id = None
    wandb_resume = "never"
    
    if resume_from_checkpoint and latest_checkpoint_path.exists():
        wandb_run_id = get_wandb_run_id(latest_checkpoint_path)
        if wandb_run_id:
            wandb_resume = "allow"
            accelerator.print(f"Found WandB run ID: {wandb_run_id}")
    
    # Initialize WandB
    wandb_init_kwargs = {
        "entity": wandb_entity,
        "name": run_name,
        "resume": wandb_resume
    }
    
    if wandb_run_id:
        wandb_init_kwargs["id"] = wandb_run_id
    
    accelerator.init_trackers(
        project_name=wandb_project,
        config={
            'dataset': dataset,
            'classifier': classifier,
            'augmentations': augmentations,
            'color_transfer_method': color_transfer_method,
            'color_transfer_prob': color_transfer_prob,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'seed': seed,
        },
        init_kwargs={"wandb": wandb_init_kwargs},
    )
    
    # Set random seed
    accelerator.print(f"Setting random seed: {seed}")
    g = random_seed(seed_value=seed, use_cuda=use_cuda)
    
    # Load color transfer method first (before dataloaders) to get native image size
    color_transfer_model = None
    color_transfer_transform = None
    use_color_transfer = 'color_transfer' in augmentations
    post_color_transfer_transform = None
    color_transfer_native_size = None
    
    if use_color_transfer:
        if color_transfer_method is None:
            raise ValueError("color_transfer_method must be specified when using color_transfer augmentation")
        if color_transfer_weights is None:
            raise ValueError("color_transfer_weights must be specified when using color_transfer augmentation")
        
        accelerator.print(f"Loading color transfer method: {color_transfer_method}")
        
        # Load the method and the underlying model
        color_transfer_transform, color_transfer_model = create_color_transfer_method(
            method_name=color_transfer_method,
            pretrained_weights=color_transfer_weights
        )
        
        # Get native image size from the color transfer method
        if hasattr(color_transfer_transform, 'get_native_image_size'):
            color_transfer_native_size = color_transfer_transform.get_native_image_size()
            accelerator.print(f"Using native image size for color transfer: {color_transfer_native_size}")
        else:
            accelerator.print(f"Warning: Color transfer method does not have get_native_image_size(), using input_size: {input_size}")
            color_transfer_native_size = input_size
                
        # Build post-color-transfer augmentations (e.g., RandomErasing)
        post_transforms = []
        if 'random_erasing' in augmentations:
            post_transforms.append(v2.RandomErasing())
        
        if post_transforms:
            post_color_transfer_transform = v2.Compose(post_transforms)
        
        accelerator.print(f"✓ Color transfer method loaded")
    
    # Prepare dataloaders (after color transfer method is loaded to get native size)
    accelerator.print(f"Loading dataset: {dataset}")

    # Scale batch size by number of processes to ensure global batch size matches argument
    per_device_batch_size = batch_size // accelerator.num_processes
    accelerator.print(f"Global batch size: {batch_size}, Per-device batch size: {per_device_batch_size} (across {accelerator.num_processes} processes)")

    train_loader, val_loader = prepare_dataloaders(
        dataset=dataset,
        data_path=data_path,
        input_size=input_size,
        batch_size=per_device_batch_size,
        num_workers=num_workers,
        augmentations=augmentations,
        g=g,
        color_transfer_native_size=color_transfer_native_size,
        accelerator=accelerator,
        **kwargs
    )
    
    # Create model
    accelerator.print(f"Creating model: {classifier}")
    num_classes = NUM_CLASSES[dataset]
    model = timm.create_model(classifier, pretrained=False, num_classes=num_classes)
    model.requires_grad_(True)
    
    # Create optimizer and scheduler
    accelerator.print("Creating optimizer and scheduler")
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=lr,
        num_epochs=epochs
    )
    
    # Create loss function based on task type
    task_type = TASK_TYPE[dataset]
    if task_type == "multi-label":
        loss_criterion = nn.BCEWithLogitsLoss()
        accelerator.print(f"Using BCEWithLogitsLoss for multi-label task")
    else:
        loss_criterion = nn.CrossEntropyLoss()
        accelerator.print(f"Using CrossEntropyLoss for {task_type} task")
    
    # Prepare for distributed training
    if color_transfer_model is not None:
        model, loss_criterion, optimizer, lr_scheduler, train_loader, val_loader, color_transfer_model = accelerator.prepare(
            model, loss_criterion, optimizer, lr_scheduler, train_loader, val_loader, color_transfer_model
        )
    else:
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
                
                start_epoch = get_resume_epoch(latest_checkpoint_path)
                best_val_loss = get_best_val_loss(latest_checkpoint_path)
                epochs_no_improve = get_epochs_no_improve(latest_checkpoint_path)
                
                accelerator.print(f"Resuming from epoch {start_epoch}")
                accelerator.print(f"Best validation loss: {best_val_loss:.4f}")
                accelerator.print(f"Epochs without improvement: {epochs_no_improve}")
            except Exception as e:
                accelerator.print(f"Error loading checkpoint: {e}")
                accelerator.print("Starting training from scratch.")
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
            input_size=input_size,
            color_transfer_transform=color_transfer_transform,
            color_transfer_prob=color_transfer_prob,
            post_color_transfer_transform=post_color_transfer_transform,
            normalization_mean=NORMALIZATION_MEAN[dataset] if use_color_transfer else None,
            normalization_std=NORMALIZATION_STD[dataset] if use_color_transfer else None
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
        accelerator.log({
            "train/loss": train_metrics['loss'],
            "train/accuracy": train_metrics['accuracy'],
            "train/balanced_accuracy": train_metrics['balanced_accuracy'],
            "train/auc": train_metrics['auc'],
            "val/loss": val_metrics['loss'],
            "val/accuracy": val_metrics['accuracy'],
            "val/balanced_accuracy": val_metrics['balanced_accuracy'],
            "val/auc": val_metrics['auc'],
        }, step=epoch + 1)
        
        # Print metrics
        accelerator.print(f"\nEpoch [{epoch + 1}/{epochs}]")
        accelerator.print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                         f"Bal Acc: {train_metrics['balanced_accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        accelerator.print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                         f"Bal Acc: {val_metrics['balanced_accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Save model
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
            
            checkpoint_metadata = {
                'epoch': epoch + 1,
                'val_loss': val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }
            
            if hasattr(accelerator, 'get_tracker'):
                try:
                    wandb_tracker = accelerator.get_tracker("wandb")
                    if wandb_tracker and hasattr(wandb_tracker, 'run'):
                        checkpoint_metadata['wandb_run_id'] = wandb_tracker.run.id
                except:
                    pass
            
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
    
    final_metadata = {
        'epoch': epochs,
        'val_loss': val_metrics['loss'],
        'best_val_loss': best_val_loss,
        'training_complete': True
    }
    
    if hasattr(accelerator, 'get_tracker'):
        try:
            wandb_tracker = accelerator.get_tracker("wandb")
            if wandb_tracker and hasattr(wandb_tracker, 'run'):
                final_metadata['wandb_run_id'] = wandb_tracker.run.id
        except:
            pass
    
    save_latest_checkpoint(
        accelerator,
        latest_checkpoint_path,
        metadata=final_metadata
    )
    
    # Rename latest to final
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
        description='Train classification model with reference-based augmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_path', type=str, default='./models', help='Path to save final trained models')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='Path to save checkpoints')
    
    # Model configuration
    parser.add_argument('--classifier', type=str, default='resnet50', help='Classifier model name (from timm)')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    
    # Augmentation configuration
    parser.add_argument('--augmentations', type=str, nargs='+', default=['none'],
                       help='List of augmentations. Options: none, gray_scale, color_jitter, '
                            'auto_augment, rand_augment, trivial_augment, aug_mix, '
                            'random_resized_crop, random_horizontal_flip, random_vertical_flip, '
                            'random_rotation, targeted_augment, color_transfer')
    parser.add_argument('--color_transfer_method', type=str, default='adain', help='Reference augmentation method (required if color_transfer in augmentations)')
    parser.add_argument('--color_transfer_weights', type=str, default=None,
                       help='Path to pretrained method weights (required if color_transfer in augmentations)')
    parser.add_argument('--color_transfer_prob', type=float, default=0.5, 
                       help='Probability of applying reference transform')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--early_stopping', type=int, default=10, help='Epochs without improvement before stopping')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                       help='Gradient accumulation steps')
    
    # Optimizer configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # System configuration
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # Checkpoint configuration
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--save_checkpoint_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    # WandB configuration
    parser.add_argument('--use_wandb', action='store_true', help='Enable WandB tracking')
    parser.add_argument('--wandb_project', type=str, default='reference-augmentation-training', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='ofu-xai', help='WandB entity name')
    parser.add_argument('--wandb_path', type=str, default=None, help='Directory for WandB logs')
    
    args = parser.parse_args()
    
    # Validate arguments
    if 'color_transfer' in args.augmentations:
        if args.color_transfer_method is None:
            parser.error("--color_transfer_method is required when using color_transfer augmentation")
        if args.color_transfer_weights is None:
            parser.error("--color_transfer_weights is required when using color_transfer augmentation")
    
    # Start training
    train(
        dataset=args.dataset,
        data_path=args.data_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        classifier=args.classifier,
        input_size=args.input_size,
        augmentations=args.augmentations,
        color_transfer_method=args.color_transfer_method,
        color_transfer_weights=args.color_transfer_weights,
        color_transfer_prob=args.color_transfer_prob,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        use_cuda=args.use_cuda,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_path=args.wandb_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        save_checkpoint_every=args.save_checkpoint_every,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
