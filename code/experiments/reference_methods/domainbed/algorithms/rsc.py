"""
RSC: Representation Self-Challenging

RSC discards the dominant features activated on the training data
to force the model to learn from remaining features, improving generalization.

Reference:
    Huang et al., "Self-Challenging Improves Cross-Domain Generalization", ECCV 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from typing import Dict, Any

from code.experiments.reference_methods.domainbed.algorithms.base import Algorithm


class RSC(Algorithm):
    """
    Representation Self-Challenging (RSC) algorithm.
    
    During training, RSC identifies and masks the most discriminative features
    based on their gradients with respect to the loss. This forces the model
    to learn from less dominant features, improving robustness.
    
    Following the official DomainBed implementation exactly.
    
    Hyperparameters:
        rsc_f_drop_factor: Fraction of features to drop (default: 0.33)
        rsc_b_drop_factor: Fraction of batch samples to apply dropping (default: 0.33)
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
        
        # drop_f and drop_b are percentiles (inverted from factor)
        # e.g., factor=0.33 means drop 33%, so keep top (1-0.33)*100 = 67th percentile
        self.drop_f = (1 - hparams.get('rsc_f_drop_factor', 0.33)) * 100
        self.drop_b = (1 - hparams.get('rsc_b_drop_factor', 0.33)) * 100
        self.num_classes = num_classes
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        RSC training step with feature masking.
        
        Following the official DomainBed implementation (Equations 1-5 from paper).
        
        Args:
            x: Input images
            y: Ground truth labels
            
        Returns:
            Dictionary with 'loss' key
        """
        device = x.device
        
        # Handle labels
        if self.task_type == "multi-label":
            all_y = y.argmax(dim=1) if y.dim() > 1 else y
        else:
            all_y = y.squeeze().long()
        
        # One-hot encode labels
        all_o = F.one_hot(all_y, self.num_classes).float()
        
        # Get features (need gradients for Equation 1)
        all_f = self.featurizer(x)
        all_f.requires_grad_(True)
        
        # Get predictions
        all_p = self.classifier(all_f)
        
        # Equation (1): compute gradients with respect to representation
        # Using the one-hot weighted predictions sum
        all_g = autograd.grad((all_p * all_o).sum(), all_f, create_graph=False)[0]
        
        # Equation (2): compute top-gradient-percentile mask
        # Higher gradient = more important = should be masked
        percentiles = np.percentile(all_g.detach().cpu().numpy(), self.drop_f, axis=1)
        percentiles = torch.tensor(percentiles, device=device, dtype=all_g.dtype)
        percentiles = percentiles.unsqueeze(1).expand_as(all_g)
        mask_f = all_g.lt(percentiles).float()
        
        # Equation (3): mute top-gradient-percentile activations
        all_f_muted = all_f * mask_f
        
        # Equation (4): compute muted predictions
        all_p_muted = self.classifier(all_f_muted)
        
        # Section 3.3: Batch Percentage
        # Compare softmax outputs to determine which samples changed most
        all_s = F.softmax(all_p, dim=1)
        all_s_muted = F.softmax(all_p_muted, dim=1)
        
        # Change = reduction in correct class probability
        changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
        percentile = np.percentile(changes.detach().cpu().numpy(), self.drop_b)
        
        # Mask samples that changed less than threshold (keep more affected samples)
        mask_b = changes.lt(percentile).float().view(-1, 1)
        
        # Combined mask: use feature mask OR batch mask
        mask = torch.logical_or(mask_f.bool(), mask_b.bool()).float()
        
        # Equations (3) and (4) again, this time muting over examples
        # Need fresh forward pass
        all_f_new = self.featurizer(x)
        all_p_muted_again = self.classifier(all_f_new * mask)
        
        # Equation (5): update with muted predictions
        loss = self.compute_loss(all_p_muted_again, y)
        
        return {'loss': loss}
