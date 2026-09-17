"""
SelfReg: Self-supervised Contrastive Regularization

SelfReg uses a contrastive-like regularization by comparing feature representations
within same-class groups, encouraging consistent representations.

Reference:
    Kim et al., "SelfReg: Self-supervised Contrastive Regularization for Domain Generalization", ICCV 2021

Note: This implementation follows the official DomainBed implementation which:
- Clusters and orders features by class label
- Shuffles within same-class groups
- Uses MSE loss between original and shuffled representations
- Applies mixup between shuffled features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from code.experiments.reference_methods.domainbed.algorithms.base import Algorithm


class SelfReg(Algorithm):
    """
    Self-supervised Contrastive Regularization (SelfReg) algorithm.
    
    Uses within-class shuffling and MSE regularization to encourage
    consistent feature representations across samples of the same class.
    
    Note: This algorithm does not use any hyperparameters in the official
    DomainBed implementation - all values are hardcoded (lam ~ Beta(0.5, 0.5),
    feature weight = 0.3).
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
        
        self.num_classes = num_classes
        self.MSEloss = nn.MSELoss()
        
        # Get feature dimension from featurizer
        if hasattr(featurizer, 'n_outputs'):
            input_feat_size = featurizer.n_outputs
        else:
            input_feat_size = 2048  # Default for ResNet50
        
        # CDPL (Contrastive Domain-invariant Projection Layer)
        # Following official implementation
        hidden_size = input_feat_size if input_feat_size == 2048 else input_feat_size * 2
        
        self.cdpl = nn.Sequential(
            nn.Linear(input_feat_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, input_feat_size),
            nn.BatchNorm1d(input_feat_size)
        )
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        SelfReg training step with within-class contrastive regularization.
        
        Following the official DomainBed implementation exactly.
        
        Args:
            x: Input images
            y: Ground truth labels
            
        Returns:
            Dictionary with 'loss' key
        """
        # Sample mixup coefficient (hardcoded in official implementation)
        lam = np.random.beta(0.5, 0.5)
        
        batch_size = y.size(0)
        
        # Handle multi-label case
        if self.task_type == "multi-label":
            # For multi-label, use argmax of labels for grouping
            y_for_sort = y.argmax(dim=1) if y.dim() > 1 else y
        else:
            y_for_sort = y.squeeze().long()
        
        # Cluster and order features into same-class groups
        with torch.no_grad():
            sorted_y, indices = torch.sort(y_for_sort)
            sorted_x = torch.zeros_like(x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = x[order]
            
            # Find class boundaries
            intervals = []
            ex = sorted_y[0].item() if sorted_y.numel() > 0 else 0
            for idx, val in enumerate(sorted_y):
                if ex != val.item():
                    intervals.append(idx)
                    ex = val.item()
            intervals.append(batch_size)
            
            # Also sort labels
            all_x = sorted_x
            all_y = y[indices]
        
        # Forward pass
        feat = self.featurizer(all_x)
        proj = self.cdpl(feat)
        output = self.classifier(feat)
        
        # Shuffle within same-class groups
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        
        ex = 0
        for end in intervals:
            if end <= ex:
                continue
            # Random permutation within class group
            group_size = end - ex
            shuffle_indices = torch.randperm(group_size, device=x.device) + ex
            shuffle_indices2 = torch.randperm(group_size, device=x.device) + ex
            
            for idx in range(group_size):
                output_2[idx + ex] = output[shuffle_indices[idx]]
                feat_2[idx + ex] = proj[shuffle_indices[idx]]
                output_3[idx + ex] = output[shuffle_indices2[idx]]
                feat_3[idx + ex] = proj[shuffle_indices2[idx]]
            ex = end
        
        # Mixup between two shuffled versions
        output_3 = lam * output_2 + (1 - lam) * output_3
        feat_3 = lam * feat_2 + (1 - lam) * feat_3
        
        # Regularization losses (hardcoded 0.3 weight in official implementation)
        L_ind_logit = self.MSEloss(output, output_2)
        L_hdl_logit = self.MSEloss(output, output_3)
        L_ind_feat = 0.3 * self.MSEloss(proj, feat_2)
        L_hdl_feat = 0.3 * self.MSEloss(proj, feat_3)
        
        # Classification loss
        cl_loss = self.compute_loss(output, all_y)
        
        # Scale regularization by classification loss (capped at 1.0)
        C_scale = min(cl_loss.item(), 1.0)
        
        # Total loss
        loss = cl_loss + C_scale * (
            lam * (L_ind_logit + L_ind_feat) + 
            (1 - lam) * (L_hdl_logit + L_hdl_feat)
        )
        
        return {'loss': loss}
