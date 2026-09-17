"""
Base Algorithm class for DomainBed methods.

All domain generalization algorithms inherit from this base class.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class Algorithm(nn.Module):
    """
    Base class for domain generalization algorithms.
    
    Subclasses should implement the update() method to define their training step.
    The predict() method is used for inference.
    
    Args:
        featurizer: Feature extractor network (backbone)
        classifier: Classification head
        num_classes: Number of output classes
        task_type: Task type ('multi-class' or 'multi-label')
        hparams: Algorithm-specific hyperparameters
    """
    
    def __init__(
        self,
        featurizer: nn.Module,
        classifier: nn.Module,
        num_classes: int,
        task_type: str,
        hparams: Dict[str, Any]
    ):
        super().__init__()
        
        self.featurizer = featurizer
        self.classifier = classifier
        self.num_classes = num_classes
        self.task_type = task_type
        self.hparams = hparams
        
        # Combined network for easy state dict access
        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        # Loss function based on task type
        if task_type == "multi-label":
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the classification loss.
        
        Handles multi-label vs multi-class automatically based on task_type.
        
        Args:
            logits: Model output logits
            y: Ground truth labels
            
        Returns:
            Loss tensor
        """
        if self.task_type == "multi-label":
            y_float = y.to(torch.float32)
            return self.loss_fn(logits, y_float)
        else:
            y_squeezed = y.squeeze().long()
            return self.loss_fn(logits, y_squeezed)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform one training step.
        
        Args:
            x: Input images (batch)
            y: Ground truth labels
            
        Returns:
            Dictionary containing at least 'loss' key with the loss tensor.
            May contain additional keys for auxiliary losses to log.
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for input images.
        
        Args:
            x: Input images
            
        Returns:
            Logits (not softmax/sigmoid - that's applied in metrics computation)
        """
        return self.network(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (alias for predict)."""
        return self.predict(x)
