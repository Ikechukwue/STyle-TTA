"""
xAILab Bamberg
University of Bamberg

@description:
Training script for DomainBed domain generalization methods.
Supports domain-agnostic methods that do not require domain labels.

Supported algorithms:
- ERM: Empirical Risk Minimization (baseline)
- Mixup: Input mixup augmentation
- RSC: Representation Self-Challenging
- SD: Spectral Decoupling
- SelfReg: Self-supervised Contrastive Regularization
- IB_ERM: Information Bottleneck ERM
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
from torchvision.transforms import v2
from accelerate import Accelerator
from accelerate.utils import tqdm
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler, create_scheduler_v2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from medmnistc.augmentation import AugMedMNISTC
from medmnistc.corruptions.registry import CORRUPTIONS_DS

# Import project modules
from code.experiments.data import create_dataset, NORMALIZATION_MEAN, NORMALIZATION_STD, NUM_CLASSES, TASK_TYPE
from code.experiments.utils.reproducibility import random_seed, worker_seed
from code.experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from code.experiments.utils.training import (
    calculate_passed_time, 
    get_wandb_run_id, 
    get_best_val_loss, 
    get_epochs_no_improve, 
    save_latest_checkpoint, 
    rename_latest_to_final, 
    get_resume_epoch
)

# Import DomainBed modules
from code.experiments.reference_methods.domainbed.algorithms import ALGORITHMS, get_algorithm
from code.experiments.reference_methods.domainbed.networks import create_featurizer_classifier, get_network_state_dict_for_timm
from code.experiments.reference_methods.domainbed.hparams import get_hparams, parse_hparams_string


def build_augmentation_transforms(
    augmentations: List[str],
    input_size: int,
    dataset: str,
) -> List:
    """
    Build augmentation transform list from augmentation names.
    
    Args:
        augmentations: List of augmentation names to apply
        input_size: Target input size for the model
        dataset: Dataset name (for targeted augmentation)
        
    Returns:
        List of transforms (not yet composed)
    """
    transforms = []
    
    for aug in augmentations:
        if aug == 'none':
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
        **kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Build augmentation pipeline
    train_augmentations = build_augmentation_transforms(
        augmentations=augmentations,
        input_size=input_size,
        dataset=dataset,
    )
    
    # Training transform
    if 'targeted_augment' in augmentations:
        # Targeted augmentations expect PIL images
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
) -> Tuple[Optimizer, CosineLRScheduler]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        num_epochs: Number of training epochs
        
    Returns:
        Tuple of (optimizer, lr_scheduler)
    """
    optimizer = create_optimizer_v2(
        model,
        opt='adam',
        lr=lr,
    )
    
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
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = (y_pred_labels == y_true).all(axis=1).mean()
        balanced_acc = accuracy
        auc = roc_auc_score(y_true, y_pred, average='samples')
    elif num_classes == 2:
        y_true_squeezed = y_true.squeeze()
        y_pred_labels = (y_pred[:, -1] > 0.5).astype(int)
        accuracy = accuracy_score(y_true_squeezed, y_pred_labels)
        balanced_acc = balanced_accuracy_score(y_true_squeezed, y_pred_labels)
        auc = roc_auc_score(y_true_squeezed, y_pred[:, -1])
    else:
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


def save_model(accelerator: Accelerator, algorithm, final_model_path: Path, classifier_name: str, num_classes: int):
    """
    Save model weights in a format compatible with timm.
    
    Args:
        accelerator: Accelerator for distributed training
        algorithm: DomainBed algorithm with network attribute
        final_model_path: Path to save the model
        classifier_name: Name of the timm model architecture
        num_classes: Number of output classes
    """
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Unwrap the model if using DDP
        unwrapped_algorithm = accelerator.unwrap_model(algorithm)
        
        # Convert to timm-compatible state dict
        timm_state_dict = get_network_state_dict_for_timm(
            unwrapped_algorithm.network,
            classifier_name,
            num_classes
        )
        
        # Save the state dict
        torch.save(timm_state_dict, final_model_path)
        accelerator.print(f"Model saved to {final_model_path}")


def train_one_epoch(
    algorithm: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: CosineLRScheduler,
    accelerator: Accelerator,
    epoch: int,
    num_epochs: int,
    num_classes: int,
    task_type: str,
) -> Dict[str, float]:
    """
    Train for one epoch using DomainBed algorithm.
    
    Args:
        algorithm: DomainBed algorithm instance
        train_loader: Training dataloader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        accelerator: Accelerator for distributed training
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        num_classes: Number of classes
        task_type: Task type for metrics
        
    Returns:
        Dictionary with training metrics
    """
    algorithm.train()
    
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
    auxiliary_losses = {}  # Track algorithm-specific losses
    y_true_list = []
    y_pred_list = []
    prediction_fn = nn.Sigmoid() if task_type == "multi-label" else nn.Softmax(dim=1)
    
    # Unwrap algorithm for accessing custom methods (update, predict)
    # DDP wraps the module, so we need to access the underlying algorithm
    unwrapped_algorithm = accelerator.unwrap_model(algorithm)
    
    num_updates = epoch * len(train_loader)
    
    for batch_idx, (x, y) in enumerate(train_loader):
        with accelerator.accumulate(algorithm):
            # Algorithm update step - use unwrapped model for custom method
            loss_dict = unwrapped_algorithm.update(x, y)
            loss = loss_dict['loss']
            
            # Check for NaNs in loss
            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"Warning: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}. Skipping.")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(algorithm.parameters(), 1.0)
            
            optimizer.step()
            lr_scheduler.step_update(num_updates=num_updates)
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += loss.item()
        
        # Track auxiliary losses
        for key, value in loss_dict.items():
            if key != 'loss':
                if key not in auxiliary_losses:
                    auxiliary_losses[key] = 0.0
                auxiliary_losses[key] += value.item() if torch.is_tensor(value) else value
        
        y_true_list.append(y.cpu())
        
        # Get predictions for metrics
        with torch.no_grad():
            outputs = unwrapped_algorithm.predict(x)
            preds = prediction_fn(outputs).detach().cpu()
            if torch.isnan(preds).any():
                preds = torch.nan_to_num(preds, nan=1.0/num_classes)
            y_pred_list.append(preds)
        
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
    
    # Add averaged auxiliary losses
    for key, value in auxiliary_losses.items():
        metrics[key] = value / len(train_loader)
    
    return metrics


def validate_one_epoch(
    algorithm: nn.Module,
    val_loader: DataLoader,
    accelerator: Accelerator,
    epoch: int,
    num_epochs: int,
    num_classes: int,
    task_type: str,
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        algorithm: DomainBed algorithm instance
        val_loader: Validation dataloader
        accelerator: Accelerator for distributed training
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        num_classes: Number of classes
        task_type: Task type for metrics
        
    Returns:
        Dictionary with validation metrics
    """
    algorithm.eval()
    
    pbar = tqdm(
        total=len(val_loader),
        bar_format="{l_bar}{bar}",
        ncols=80,
        initial=0,
        position=0,
        leave=False,
    )
    pbar.set_description(f"Val [{epoch + 1}/{num_epochs}]")
    
    # Create loss function for validation
    if task_type == "multi-label":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    prediction_fn = nn.Sigmoid() if task_type == "multi-label" else nn.Softmax(dim=1)
    
    # Unwrap algorithm for accessing custom methods (predict)
    unwrapped_algorithm = accelerator.unwrap_model(algorithm)
    
    with torch.no_grad():
        for x, y in val_loader:
            # Forward pass
            outputs = unwrapped_algorithm.predict(x)
            
            # Compute loss
            if task_type == "multi-label":
                y_float = y.to(torch.float32)
                loss = loss_fn(outputs, y_float)
            else:
                y_squeezed = y.squeeze().long()
                loss = loss_fn(outputs, y_squeezed)
            
            # Gather metrics across processes
            gathered_outputs = accelerator.gather_for_metrics(outputs)
            gathered_y = accelerator.gather_for_metrics(y)
            reduced_loss = accelerator.reduce(loss, reduction="mean")
            
            total_loss += reduced_loss.item()
            y_true_list.append(gathered_y.cpu())
            
            preds = prediction_fn(gathered_outputs).cpu()
            if torch.isnan(preds).any():
                preds = torch.nan_to_num(preds, nan=1.0/num_classes)
            y_pred_list.append(preds)
            
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
    domainbed_algorithm: str,
    hparams: Optional[Dict[str, Any]],
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
    Main training function for DomainBed algorithms.
    
    Args:
        dataset: Dataset name
        data_path: Path to dataset
        output_path: Path to save final trained models
        checkpoint_path: Path to save checkpoints for resuming training
        classifier: Classifier model name (from timm)
        input_size: Input image size
        augmentations: List of augmentation names
        domainbed_algorithm: Name of DomainBed algorithm
        hparams: Algorithm-specific hyperparameters
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

    # Create run name and paths
    run_name = f"{dataset}-{classifier}-{domainbed_algorithm}-seed{seed}"
    latest_checkpoint_path = checkpoint_path / f"{run_name}_latest"
    final_model_path = output_path / f"{run_name}.pth"
    
    # Set WandB directory
    if wandb_path is not None:
        os.environ['WANDB_DIR'] = wandb_path
        Path(wandb_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize accelerator
    log_with = ["wandb"] if use_wandb else None
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=log_with
    )
    
    # Check for checkpoint resumption
    wandb_run_id = None
    wandb_resume = "never"
    
    if resume_from_checkpoint and latest_checkpoint_path.exists():
        wandb_run_id = get_wandb_run_id(latest_checkpoint_path)
        if wandb_run_id:
            wandb_resume = "allow"
            accelerator.print(f"Found WandB run ID: {wandb_run_id}")
    
    # Get algorithm hyperparameters
    algorithm_hparams = get_hparams(domainbed_algorithm, hparams)
    
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
            'domainbed_algorithm': domainbed_algorithm,
            'augmentations': augmentations,
            'hparams': algorithm_hparams,
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
    
    # Prepare dataloaders
    accelerator.print(f"Loading dataset: {dataset}")
    
    per_device_batch_size = batch_size // accelerator.num_processes
    accelerator.print(f"Global batch size: {batch_size}, Per-device batch size: {per_device_batch_size}")
    
    train_loader, val_loader = prepare_dataloaders(
        dataset=dataset,
        data_path=data_path,
        input_size=input_size,
        batch_size=per_device_batch_size,
        num_workers=num_workers,
        augmentations=augmentations,
        g=g,
        accelerator=accelerator,
        **kwargs
    )
    
    # Create model components
    accelerator.print(f"Creating model: {classifier}")
    num_classes = NUM_CLASSES[dataset]
    task_type = TASK_TYPE[dataset]
    
    featurizer, classifier_head, network = create_featurizer_classifier(
        classifier_name=classifier,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Create DomainBed algorithm
    accelerator.print(f"Creating algorithm: {domainbed_algorithm}")
    accelerator.print(f"Hyperparameters: {algorithm_hparams}")
    
    algorithm = get_algorithm(
        name=domainbed_algorithm,
        featurizer=featurizer,
        classifier=classifier_head,
        num_classes=num_classes,
        task_type=task_type,
        hparams=algorithm_hparams
    )
    
    # Create optimizer and scheduler
    accelerator.print("Creating optimizer and scheduler")
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model=algorithm,
        lr=lr,
        num_epochs=epochs
    )
    
    # Prepare for distributed training
    algorithm, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        algorithm, optimizer, lr_scheduler, train_loader, val_loader
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
            algorithm=algorithm,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            epoch=epoch,
            num_epochs=epochs,
            num_classes=num_classes,
            task_type=task_type,
        )
        
        # Validate
        val_metrics = validate_one_epoch(
            algorithm=algorithm,
            val_loader=val_loader,
            accelerator=accelerator,
            epoch=epoch,
            num_epochs=epochs,
            num_classes=num_classes,
            task_type=task_type,
        )
        
        # Log metrics
        log_dict = {
            "train/loss": train_metrics['loss'],
            "train/accuracy": train_metrics['accuracy'],
            "train/balanced_accuracy": train_metrics['balanced_accuracy'],
            "train/auc": train_metrics['auc'],
            "val/loss": val_metrics['loss'],
            "val/accuracy": val_metrics['accuracy'],
            "val/balanced_accuracy": val_metrics['balanced_accuracy'],
            "val/auc": val_metrics['auc'],
        }
        
        # Add algorithm-specific losses
        for key in train_metrics:
            if key not in ['loss', 'accuracy', 'balanced_accuracy', 'auc']:
                log_dict[f"train/{key}"] = train_metrics[key]
        
        accelerator.log(log_dict, step=epoch + 1)
        
        # Print metrics
        accelerator.print(f"\nEpoch [{epoch + 1}/{epochs}]")
        accelerator.print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                         f"Bal Acc: {train_metrics['balanced_accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
        accelerator.print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                         f"Bal Acc: {val_metrics['balanced_accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_metrics['loss']
            save_model(accelerator, algorithm, final_model_path, classifier, num_classes)
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
        description='Train classification model with DomainBed algorithms',
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
    
    # Algorithm configuration
    parser.add_argument('--domainbed_algorithm', type=str, default='ERM',
                       choices=list(ALGORITHMS.keys()),
                       help=f'DomainBed algorithm. Options: {list(ALGORITHMS.keys())}')
    parser.add_argument('--hparams', type=str, default=None,
                       help='Algorithm hyperparameters as JSON string (e.g., \'{"mixup_alpha": 0.4}\')')
    
    # Augmentation configuration
    parser.add_argument('--augmentations', type=str, nargs='+', default=['none'],
                       help='List of augmentations. Options: none, gray_scale, color_jitter, '
                            'auto_augment, rand_augment, trivial_augment, aug_mix, '
                            'random_resized_crop, random_flip, random_erasing, targeted_augment')
    
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
    parser.add_argument('--wandb_project', type=str, default='domainbed-training', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default='ofu-xai', help='WandB entity name')
    parser.add_argument('--wandb_path', type=str, default=None, help='Directory for WandB logs')
    
    args = parser.parse_args()
    
    # Parse hyperparameters
    hparams = parse_hparams_string(args.hparams)
    
    # Start training
    train(
        dataset=args.dataset,
        data_path=args.data_path,
        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        classifier=args.classifier,
        input_size=args.input_size,
        augmentations=args.augmentations,
        domainbed_algorithm=args.domainbed_algorithm,
        hparams=hparams,
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
