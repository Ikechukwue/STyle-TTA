"""
MMLD: Domain Generalization Using a Mixture of Multiple Latent Domains
Paper: "Domain Generalization Using a Mixture of Multiple Latent Domains" (AAAI 2020)
Authors: Matsuura and Harada

This implementation adapts MMLD for single-domain generalization by:
1. Creating pseudo-domains via augmentation or clustering
2. Using domain discriminator with gradient reversal layer (GRL)
3. Entropy regularization to prevent overconfident predictions
4. Progressive GRL weight scheduling

Key components:
- GradientReversalLayer (GRL) for adversarial domain alignment
- Domain discriminator to classify pseudo-domains
- Entropy loss for regularization
- Progressive scheduling: alpha = 2/(1+exp(-10*p)) - 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Dict, Any, Optional
import random
import math

from .base import Algorithm


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) for adversarial training.
    Forward: identity
    Backward: negate gradients and scale by lambda
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * -ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal."""
    return GradientReversalFunction.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator with optional gradient reversal.
    Predicts which pseudo-domain a sample belongs to.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 1024,
        num_domains: int = 3,
        use_grl: bool = True
    ):
        super().__init__()
        self.use_grl = use_grl
        self.lambd = 0.0
        
        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_domains),
        )
    
    def set_lambd(self, lambd: float):
        """Set the gradient reversal weight."""
        self.lambd = lambd
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_grl:
            x = grad_reverse(x, self.lambd)
        return self.model(x)


class EntropyLoss(nn.Module):
    """
    Entropy loss to regularize classifier predictions.
    Encourages less confident predictions for better generalization.
    
    H(p) = -sum(p * log(p))
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Logits [B, C]
        Returns:
            Mean entropy loss
        """
        p = F.softmax(x, dim=1)
        log_p = F.log_softmax(x, dim=1)
        entropy = -1.0 * (p * log_p).sum(dim=1).mean()
        return entropy


def create_pseudo_domains(
    x: torch.Tensor,
    num_domains: int = 3
) -> tuple:
    """
    Create pseudo-domains via augmentations.
    
    Args:
        x: Input images [B, C, H, W]
        num_domains: Number of pseudo-domains to create
        
    Returns:
        Augmented images, domain labels
    """
    B = x.size(0)
    device = x.device
    
    # Assign samples to pseudo-domains randomly or via simple augmentation
    domain_labels = torch.randint(0, num_domains, (B,), device=device)
    
    x_aug = x.clone()
    
    for i in range(B):
        domain = domain_labels[i].item()
        
        if domain == 0:
            # Original (no augmentation)
            pass
        elif domain == 1:
            # Color shift
            shift = random.uniform(-0.2, 0.2)
            x_aug[i] = (x_aug[i] + shift).clamp(0, 1)
        elif domain == 2:
            # Contrast adjustment
            factor = random.uniform(0.7, 1.3)
            mean = x_aug[i].mean()
            x_aug[i] = (mean + factor * (x_aug[i] - mean)).clamp(0, 1)
        elif domain == 3:
            # Brightness
            factor = random.uniform(0.8, 1.2)
            x_aug[i] = (x_aug[i] * factor).clamp(0, 1)
        elif domain == 4:
            # Saturation
            gray = x_aug[i].mean(dim=0, keepdim=True)
            factor = random.uniform(0.5, 1.5)
            x_aug[i] = (gray + factor * (x_aug[i] - gray)).clamp(0, 1)
        else:
            # Gaussian noise
            noise = torch.randn_like(x_aug[i]) * 0.1
            x_aug[i] = (x_aug[i] + noise).clamp(0, 1)
    
    return x_aug, domain_labels


class MMLD(Algorithm):
    """
    MMLD: Domain Generalization Using a Mixture of Multiple Latent Domains
    
    Uses domain discriminator with gradient reversal to learn domain-invariant
    features, combined with entropy regularization.
    
    Args:
        featurizer: Feature extraction network
        classifier: Classification head
        num_classes: Number of output classes
        task_type: 'multi-class' or 'multi-label'
        hparams: Dictionary with hyperparameters:
            - num_domains: Number of pseudo-domains (default: 3)
            - disc_hidden: Hidden dimension of discriminator (default: 1024)
            - entropy_weight: Weight for entropy loss (default: 1.0)
            - grl_weight: Maximum GRL weight (default: 1.0)
            - use_grl: Whether to use gradient reversal (default: True)
            - progressive: Use progressive GRL scheduling (default: True)
    """
    
    def __init__(
        self,
        featurizer: nn.Module,
        classifier: nn.Module,
        num_classes: int,
        task_type: str = 'multi-class',
        hparams: Optional[Dict[str, Any]] = None
    ):
        super().__init__(featurizer, classifier, num_classes, task_type, hparams)
        
        # Hyperparameters
        hparams = hparams or {}
        self.num_domains = hparams.get('num_domains', 3)
        self.disc_hidden = hparams.get('disc_hidden', 1024)
        self.entropy_weight = hparams.get('entropy_weight', 1.0)
        self.grl_weight = hparams.get('grl_weight', 1.0)
        self.use_grl = hparams.get('use_grl', True)
        self.progressive = hparams.get('progressive', True)
        
        # Get feature dimension
        if hasattr(featurizer, 'n_outputs'):
            feature_dim = featurizer.n_outputs
        else:
            feature_dim = 512  # Default
        
        # Domain discriminator
        self.discriminator = DomainDiscriminator(
            in_features=feature_dim,
            hidden_dim=self.disc_hidden,
            num_domains=self.num_domains,
            use_grl=self.use_grl
        )
        
        # Losses
        if self.task_type == 'multi-class':
            self.class_criterion = nn.CrossEntropyLoss()
        else:
            self.class_criterion = nn.BCEWithLogitsLoss()
        
        self.domain_criterion = nn.CrossEntropyLoss()
        self.entropy_criterion = EntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 100  # Will be updated during training
    
    def set_epoch(self, epoch: int, total_epochs: int):
        """Set current epoch for progressive scheduling."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def get_grl_weight(self) -> float:
        """Get current GRL weight based on progressive schedule."""
        if not self.progressive:
            return self.grl_weight
        
        p = self.current_epoch / max(self.total_epochs, 1)
        alpha = (2.0 / (1.0 + math.exp(-10 * p)) - 1) * self.grl_weight
        return alpha
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features and handle spatial dimensions."""
        features = self.featurizer(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.extract_features(x)
        return self.classifier(features)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update model with MMLD training.
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B] for multi-class or [B, C] for multi-label
            
        Returns:
            Dictionary with loss values
        """
        # Create pseudo-domains via augmentation
        x_aug, domain_labels = create_pseudo_domains(x, self.num_domains)
        
        # Update GRL weight based on epoch
        alpha = self.get_grl_weight()
        self.discriminator.set_lambd(alpha)
        
        # Extract features
        features = self.extract_features(x_aug)
        
        # Classification
        logits = self.classifier(features)
        class_loss = self.class_criterion(logits, y)
        
        # Domain discrimination (with GRL)
        domain_logits = self.discriminator(features)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)
        
        # Entropy regularization
        entropy_weight = self.get_entropy_weight()
        entropy_loss = self.entropy_criterion(logits)
        
        # Total loss
        total_loss = class_loss + domain_loss + entropy_weight * entropy_loss
        
        # Compute accuracies for logging
        with torch.no_grad():
            _, pred_class = torch.max(logits, 1)
            class_acc = (pred_class == y).float().mean()
            
            _, pred_domain = torch.max(domain_logits, 1)
            domain_acc = (pred_domain == domain_labels).float().mean()
        
        return {
            'loss': total_loss,
            'class_loss': class_loss,
            'domain_loss': domain_loss,
            'entropy_loss': entropy_loss,
            'class_acc': class_acc,
            'domain_acc': domain_acc,
            'grl_weight': torch.tensor(alpha, device=x.device),
        }
    
    def get_entropy_weight(self) -> float:
        """Get current entropy weight based on progressive schedule."""
        if not self.progressive:
            return self.entropy_weight
        
        p = self.current_epoch / max(self.total_epochs, 1)
        beta = (2.0 / (1.0 + math.exp(-10 * p)) - 1) * self.entropy_weight
        return beta
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits for input samples.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Class logits [B, num_classes]
        """
        return self.forward(x)
