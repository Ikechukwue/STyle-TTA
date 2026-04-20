"""
SD: Spectral Decoupling

Spectral Decoupling adds an L2 penalty on the classifier's output logits,
encouraging the model to produce smaller, more robust predictions.

Reference:
    Pezeshki et al., "Gradient Starvation: A Learning Proclivity in Neural Networks", NeurIPS 2021
"""

import torch
from typing import Dict, Any

from experiments.reference_methods.domainbed.algorithms.base import Algorithm


class SD(Algorithm):
    """
    Spectral Decoupling (SD) algorithm.
    
    Adds an L2 regularization term on the output logits to prevent
    gradient starvation and encourage more balanced feature learning.
    
    Hyperparameters:
        sd_reg: Regularization strength (default: 0.1)
    """
    
    def __init__(
        self,
        featurizer,
        classifier,
        num_classes: int,
        task_type: str,
        hparams: Dict[str, Any]
    ):
        super().__init__(
            featurizer=featurizer,
            classifier=classifier,
            num_classes=num_classes,
            task_type=task_type,
            hparams=hparams
        )
        
        self.sd_reg = hparams.get('sd_reg', 0.1)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        SD training step with logit penalty.
        
        Args:
            x: Input images
            y: Ground truth labels
            
        Returns:
            Dictionary with 'loss', 'cls_loss', and 'sd_penalty' keys
        """
        # Forward pass
        features = self.featurizer(x)
        logits = self.classifier(features)
        
        # Classification loss
        cls_loss = self.compute_loss(logits, y)
        
        # Spectral decoupling penalty: L2 norm of logits
        sd_penalty = (logits ** 2).mean()
        
        # Total loss
        loss = cls_loss + self.sd_reg * sd_penalty
        
        return {
            'loss': loss,
            'cls_loss': cls_loss,
            'sd_penalty': sd_penalty
        }
