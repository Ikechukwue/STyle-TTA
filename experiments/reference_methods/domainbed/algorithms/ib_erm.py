"""
IB_ERM: Information Bottleneck ERM

IB_ERM adds a variance penalty to the features, encouraging the model
to learn compact representations by minimizing the information in the
feature space while maintaining classification accuracy.

Reference:
    Ahuja et al., "Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization", NeurIPS 2021
"""

import torch
from typing import Dict, Any

from experiments.reference_methods.domainbed.algorithms.base import Algorithm


class IB_ERM(Algorithm):
    """
    Information Bottleneck ERM (IB_ERM) algorithm.
    
    Adds a variance penalty to encourage compact feature representations,
    implementing the information bottleneck principle.
    
    Following the official DomainBed implementation which includes:
    - Penalty annealing (no penalty until ib_penalty_anneal_iters)
    - Feature variance penalty
    
    Hyperparameters:
        ib_lambda: Weight for variance penalty (default: 1e-3)
        ib_penalty_anneal_iters: Number of iterations before applying penalty (default: 500)
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
        
        self.ib_lambda = hparams.get('ib_lambda', 1e-3)
        self.ib_penalty_anneal_iters = hparams.get('ib_penalty_anneal_iters', 500)
        
        # Track update count for annealing
        self.register_buffer('update_count', torch.tensor([0]))
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        IB_ERM training step with variance penalty.
        
        Following the official DomainBed implementation.
        
        Args:
            x: Input images
            y: Ground truth labels
            
        Returns:
            Dictionary with 'loss', 'nll', and 'ib_penalty' keys
        """
        # Determine penalty weight based on annealing schedule
        if self.update_count >= self.ib_penalty_anneal_iters:
            ib_penalty_weight = self.ib_lambda
        else:
            ib_penalty_weight = 0.0
        
        # Forward pass
        features = self.featurizer(x)
        logits = self.classifier(features)
        
        # Classification loss (NLL)
        nll = self.compute_loss(logits, y)
        
        # Information bottleneck penalty: variance of features
        # This encourages compact representations
        ib_penalty = features.var(dim=0).mean()
        
        # Total loss
        loss = nll + ib_penalty_weight * ib_penalty
        
        # Increment update count
        self.update_count += 1
        
        return {
            'loss': loss,
            'nll': nll,
            'ib_penalty': ib_penalty
        }
