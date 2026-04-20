"""
Mixup: Input Mixup for Domain Generalization

Mixup creates virtual training examples by linearly interpolating
random pairs of samples and their labels within a batch.

Reference:
    Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
"""

import torch
import numpy as np
from typing import Dict, Any

from experiments.reference_methods.domainbed.algorithms.base import Algorithm


class Mixup(Algorithm):
    """
    Mixup algorithm for domain generalization.
    
    Interpolates pairs of samples using coefficients drawn from Beta(alpha, alpha).
    For multi-class tasks, the loss is computed as a weighted sum of the individual
    losses. For multi-label tasks, the labels are directly interpolated.
    
    Hyperparameters:
        mixup_alpha: Parameter for Beta distribution (default: 0.2)
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
        
        self.alpha = hparams.get('mixup_alpha', 0.2)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Mixup training step.
        
        Args:
            x: Input images
            y: Ground truth labels
            
        Returns:
            Dictionary with 'loss' key
        """
        batch_size = x.size(0)
        
        # Sample mixup coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Ensure lam >= 0.5 (so first sample dominates, for label assignment)
        lam = max(lam, 1 - lam)
        
        # Random permutation for pairing
        index = torch.randperm(batch_size, device=x.device)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Forward pass on mixed inputs
        features = self.featurizer(mixed_x)
        logits = self.classifier(features)
        
        # Compute loss
        if self.task_type == "multi-label":
            # For multi-label, interpolate the labels directly
            y_float = y.to(torch.float32)
            y_mixed = lam * y_float + (1 - lam) * y_float[index]
            loss = self.loss_fn(logits, y_mixed)
        else:
            # For multi-class, compute weighted loss
            y_a = y.squeeze().long()
            y_b = y[index].squeeze().long()
            loss = lam * self.loss_fn(logits, y_a) + (1 - lam) * self.loss_fn(logits, y_b)
        
        return {'loss': loss}
