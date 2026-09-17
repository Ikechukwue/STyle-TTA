"""
ERM: Empirical Risk Minimization

Standard cross-entropy training baseline.
This is the simplest algorithm with no domain generalization regularization.
"""

import torch
from typing import Dict, Any

from code.experiments.reference_methods.domainbed.algorithms.base import Algorithm


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM) - baseline algorithm.
    
    Simply minimizes the classification loss without any domain generalization
    regularization. This serves as the baseline for comparison.
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
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Standard training step.
        
        Args:
            x: Input images
            y: Ground truth labels
            
        Returns:
            Dictionary with 'loss' key
        """
        # Forward pass
        features = self.featurizer(x)
        logits = self.classifier(features)
        
        # Compute classification loss
        loss = self.compute_loss(logits, y)
        
        return {'loss': loss}
