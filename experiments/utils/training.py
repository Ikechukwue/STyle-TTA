"""
xAILab Bamberg
University of Bamberg

@description:
Helper functions for training.
"""

# Import packages
import os
import torch.nn as nn
from accelerate import Accelerator
from typing import Tuple
from pathlib import Path
from typing import Optional
import shutil
import os
import pickle


def calculate_passed_time(start_time: float, end_time: float) -> Tuple[int, int, float]:
    """
    Calculate the passed time.

    Args:
        start_time (float): Start time.
        end_time (float): End time.

    Returns:
        (int, int, float): total duration in hours, minutes and seconds
    """

    # Calculate the duration
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Return the duration in hours, minutes and seconds
    return int(hours), int(minutes), seconds


def get_resume_iteration(checkpoint_dir: Path) -> int:
    """
    Get the iteration number from which to resume training.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        Iteration number, or 0 if cannot determine
    """
    # Try to read iteration from checkpoint metadata
    metadata_path = checkpoint_dir / "training_metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            try:
                metadata = pickle.load(f)
                return metadata.get('iteration', 0)
            except:
                pass
    
    return 0


def get_resume_epoch(checkpoint_dir: Path) -> int:
    """
    Get the epoch number from which to resume training.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        Epoch number, or 0 if cannot determine
    """
    # Try to read epoch from checkpoint metadata
    metadata_path = checkpoint_dir / "training_metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            try:
                metadata = pickle.load(f)
                return metadata.get('epoch', 0)
            except:
                pass
    
    return 0


def get_wandb_run_id(checkpoint_dir: Path) -> Optional[str]:
    """
    Get the WandB run ID from checkpoint metadata.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        WandB run ID string, or None if not found
    """
    # Try to read run ID from checkpoint metadata
    metadata_path = checkpoint_dir / "training_metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            try:
                metadata = pickle.load(f)
                return metadata.get('wandb_run_id', None)
            except:
                pass
    
    return None


def get_best_val_loss(checkpoint_dir: Path) -> float:
    """
    Get the best validation loss from checkpoint metadata.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        Best validation loss, or np.inf if not found
    """
    # Try to read best val loss from checkpoint metadata
    metadata_path = checkpoint_dir / "training_metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            try:
                metadata = pickle.load(f)
                return metadata.get('best_val_loss', float('inf'))
            except:
                pass
    
    return float('inf')


def get_epochs_no_improve(checkpoint_dir: Path) -> int:
    """
    Get the epochs without improvement count from checkpoint metadata.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        Number of epochs without improvement, or 0 if not found
    """
    # Try to read epochs_no_improve from checkpoint metadata
    metadata_path = checkpoint_dir / "training_metadata.pkl"
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            try:
                metadata = pickle.load(f)
                return metadata.get('epochs_no_improve', 0)
            except:
                pass
    
    return 0


def save_training_metadata(
    checkpoint_path: Path,
    metadata: dict
):
    """
    Save custom training metadata alongside Accelerate checkpoint.
    
    This stores additional information (like iteration/epoch number, loss values, etc.)
    that is not part of the standard Accelerate checkpoint. This metadata can be
    used to track progress or for custom resume logic.
    
    Supports both iteration-based (AdaIN) and epoch-based (AdaAttN) training:
    - For iteration-based: include 'iteration' and optionally 'total_iters'
    - For epoch-based: include 'epoch' and optionally 'total_epochs'
    
    Note: Only call this from the main process to avoid race conditions.
    
    Args:
        checkpoint_path: Directory where checkpoint is saved
        metadata: Dictionary containing metadata to save (e.g., iteration, epoch, loss, wandb_run_id)
    
    Example (iteration-based):
        save_training_metadata(
            checkpoint_path,
            {'iteration': 50000, 'total_iters': 50000, 'wandb_run_id': 'abc123'}
        )
    
    Example (epoch-based):
        save_training_metadata(
            checkpoint_path,
            {'epoch': 10, 'total_epochs': 200, 'wandb_run_id': 'abc123'}
        )
    """    
    # Ensure checkpoint directory exists
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    metadata_path = checkpoint_path / "training_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)


def save_latest_checkpoint(
    accelerator,
    checkpoint_path: Path,
    metadata: dict = None
):
    """
    Save the latest checkpoint during training (overwriting previous latest).
    
    This saves a checkpoint that can be used to resume training. Only one "latest"
    checkpoint is kept to save disk space.
    
    Supports both iteration-based (AdaIN) and epoch-based (AdaAttN) training.
    Include appropriate keys in metadata:
    - Iteration-based: 'iteration', 'total_iters'
    - Epoch-based: 'epoch', 'total_epochs'
    
    Following Accelerate best practices:
    - All processes call wait_for_everyone() and save_state()
    - Only main process saves metadata
    - Never guard save_state() with is_main_process (causes deadlocks)
    
    Args:
        accelerator: Accelerator instance
        checkpoint_path: Base checkpoint directory (e.g., /path/to/checkpoints/adain/)
        metadata: Dictionary containing metadata to save (e.g., iteration, epoch, loss, wandb_run_id)
    
    Example (iteration-based):
        save_latest_checkpoint(
            accelerator, 
            checkpoint_path,
            metadata={'iteration': 50000, 'total_iters': 100000, 'wandb_run_id': 'abc123'}
        )
    
    Example (epoch-based):
        save_latest_checkpoint(
            accelerator, 
            checkpoint_path,
            metadata={'epoch': 10, 'total_epochs': 200, 'wandb_run_id': 'abc123'}
        )
    
    References:
        - https://huggingface.co/docs/accelerate/en/usage_guides/checkpoint
    """   
    # Sync all processes before saving (critical for multi-GPU)
    accelerator.wait_for_everyone()

    # Ensure checkpoint directory exists
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save resumable checkpoint - ALL processes must participate
    accelerator.save_state(str(checkpoint_path))
    
    # Save custom metadata if provided (only main process)
    if metadata and accelerator.is_main_process:
        save_training_metadata(checkpoint_path, metadata)


def rename_latest_to_final(
    checkpoint_path: Path
):
    """
    Rename the latest checkpoint to final after training completes.
    
    This should only be called by the main process after training is complete.
    
    Args:
        checkpoint_path: Checkpoint directory to rename
        dataset_source: Source dataset name
        dataset_reference: Reference dataset name
    """
    latest_path = checkpoint_path

    if "_latest" in checkpoint_path.name:
        final_name = checkpoint_path.name.replace("_latest", "_final")
        final_path = checkpoint_path.parent / final_name
    else:
        final_path = checkpoint_path.parent / (checkpoint_path.name + "_final")
    
    if latest_path.exists():
        # Remove old final if it exists
        if final_path.exists():
            shutil.rmtree(final_path)
        
        # Rename latest to final
        latest_path.rename(final_path)
        print(f"Renamed checkpoint from 'latest' to 'final': {final_path}")
    else:
        print(f"Warning: No latest checkpoint found at {latest_path}")


def save_model(
    accelerator,
    model,
    save_path: Path
):
    """
    Save portable model weights for inference.
    
    Saves only the model state dict (weights) to a .pth file.
    Uses accelerator.get_state_dict() to properly gather distributed weights.
    
    Args:
        accelerator: Accelerator instance
        model: Model to save (will be unwrapped and gathered)
        save_path: Path to save the model weights (.pth file)
    
    Example:
        save_portable_model(accelerator, my_model, 'checkpoints/model.pth')
    
    References:
        - https://huggingface.co/docs/accelerate/en/usage_guides/checkpoint
    """

    # Let all processes finish before saving the model
    accelerator.wait_for_everyone()
    
    # Unwrap the model
    unwrapped_model = accelerator.unwrap_model(model)

    # Ensure parent directory exists
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    accelerator.save(unwrapped_model.state_dict(), str(save_path))
