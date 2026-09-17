"""
MixStyle: Domain Generalization with MixStyle
Paper: "Domain Generalization with MixStyle" (ICLR 2021)
Authors: Zhou et al.

Key idea: Probabilistically mix instance-level feature statistics (mean and 
standard deviation) of training samples across the batch. This implicitly 
synthesizes new domains at the feature level for regularizing CNN training.

The mixing coefficient λ is sampled from Beta(α, α) distribution.

Reference implementation:
https://github.com/KaiyangZhou/mixstyle-release
https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/modeling/ops/mixstyle.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import random

from .base import Algorithm


class MixStyleModule(nn.Module):
    """
    MixStyle layer that can be inserted into CNN architectures.
    
    Mixes instance-level feature statistics between samples in a batch
    to synthesize new domain styles.
    
    Args:
        p: Probability of applying MixStyle
        alpha: Parameter of the Beta distribution for mixing coefficient
        eps: Small value for numerical stability
        mix: Mixing strategy ('random' or 'crossdomain')
    """
    
    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.1,
        eps: float = 1e-6,
        mix: str = 'random'
    ):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.mix = mix
        self._activated = True
        
        # Beta distribution for sampling mixing coefficients
        self.beta = torch.distributions.Beta(alpha, alpha)
    
    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'
    
    def set_activation_status(self, status: bool = True):
        """Enable or disable MixStyle."""
        self._activated = status
    
    def update_mix_method(self, mix: str = 'random'):
        """Update mixing strategy."""
        self.mix = mix
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MixStyle to feature maps.
        
        Args:
            x: Feature maps [B, C, H, W]
            
        Returns:
            Feature maps with mixed statistics [B, C, H, W]
        """
        if not self.training or not self._activated:
            return x
        
        if random.random() > self.p:
            return x
        
        B = x.size(0)
        
        # Compute instance-level statistics
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        
        # Detach statistics (no gradient through them)
        mu, sig = mu.detach(), sig.detach()
        
        # Normalize features
        x_normed = (x - mu) / sig
        
        # Sample mixing coefficient from Beta distribution
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)
        
        # Get permutation for mixing
        if self.mix == 'random':
            perm = torch.randperm(B)
        elif self.mix == 'crossdomain':
            # Split into two halves and swap
            perm = torch.arange(B - 1, -1, -1)  # Inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError(f"Unknown mix method: {self.mix}")
        
        # Get statistics from shuffled samples
        mu2, sig2 = mu[perm], sig[perm]
        
        # Mix statistics
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        
        # Apply mixed statistics
        return x_normed * sig_mix + mu_mix


class MixStyleFeaturizer(nn.Module):
    """
    Wrapper that adds MixStyle layers to a featurizer.
    
    For single-domain generalization, we apply MixStyle after early conv layers
    to synthesize new domain styles within the same source domain.
    """
    
    def __init__(
        self,
        featurizer: nn.Module,
        p: float = 0.5,
        alpha: float = 0.1,
        mix: str = 'random',
        num_mixstyle_layers: int = 2
    ):
        super().__init__()
        self.featurizer = featurizer
        self.num_mixstyle_layers = num_mixstyle_layers
        
        # Create MixStyle modules
        self.mixstyle_layers = nn.ModuleList([
            MixStyleModule(p=p, alpha=alpha, mix=mix)
            for _ in range(num_mixstyle_layers)
        ])
        
        # Copy n_outputs from featurizer if available
        if hasattr(featurizer, 'n_outputs'):
            self.n_outputs = featurizer.n_outputs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MixStyle applied after early layers.
        
        For a generic featurizer, we apply MixStyle to intermediate features.
        """
        # Get all submodules as a list
        children = list(self.featurizer.children())
        
        if len(children) == 0:
            # If featurizer has no children (e.g., Sequential), just apply it
            return self.featurizer(x)
        
        # Apply MixStyle after first few layers
        mixstyle_idx = 0
        for i, layer in enumerate(children):
            x = layer(x)
            
            # Apply MixStyle after early layers (not the last one)
            if (mixstyle_idx < self.num_mixstyle_layers and 
                i < len(children) - 1 and 
                x.dim() == 4):  # Only for spatial features
                x = self.mixstyle_layers[mixstyle_idx](x)
                mixstyle_idx += 1
        
        return x


def apply_mixstyle_to_features(
    x: torch.Tensor,
    p: float = 0.5,
    alpha: float = 0.1,
    eps: float = 1e-6,
    training: bool = True
) -> torch.Tensor:
    """
    Functional version of MixStyle for applying to feature maps.
    
    Args:
        x: Feature maps [B, C, H, W] or [B, C]
        p: Probability of applying MixStyle
        alpha: Beta distribution parameter
        eps: Small value for numerical stability
        training: Whether in training mode
        
    Returns:
        Feature maps with mixed statistics
    """
    if not training or random.random() > p:
        return x
    
    B = x.size(0)
    
    if x.dim() == 4:
        # Spatial features [B, C, H, W]
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + eps).sqrt()
        
        lmda = torch.distributions.Beta(alpha, alpha).sample((B, 1, 1, 1))
    else:
        # Vector features [B, C]
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        sig = (var + eps).sqrt()
        
        lmda = torch.distributions.Beta(alpha, alpha).sample((B, 1))
    
    lmda = lmda.to(x.device)
    mu, sig = mu.detach(), sig.detach()
    
    x_normed = (x - mu) / sig
    
    perm = torch.randperm(B)
    mu2, sig2 = mu[perm], sig[perm]
    
    mu_mix = mu * lmda + mu2 * (1 - lmda)
    sig_mix = sig * lmda + sig2 * (1 - lmda)
    
    return x_normed * sig_mix + mu_mix


class MixStyle(Algorithm):
    """
    MixStyle: Domain Generalization with MixStyle
    
    Applies MixStyle to intermediate feature representations to synthesize
    new domain styles by mixing instance-level statistics.
    
    Args:
        featurizer: Feature extraction network
        classifier: Classification head
        num_classes: Number of output classes
        task_type: 'multi-class' or 'multi-label'
        hparams: Dictionary with hyperparameters:
            - p: Probability of applying MixStyle (default: 0.5)
            - alpha: Beta distribution parameter (default: 0.1)
            - mix: Mixing strategy 'random' or 'crossdomain' (default: 'random')
            - eps: Numerical stability constant (default: 1e-6)
            - apply_to_features: Apply MixStyle to final features (default: True)
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
        self.p = hparams.get('p', 0.5)
        self.alpha = hparams.get('alpha', 0.1)
        self.mix = hparams.get('mix', 'random')
        self.eps = hparams.get('eps', 1e-6)
        self.apply_to_features = hparams.get('apply_to_features', True)
        
        # MixStyle module for feature-level mixing
        self.mixstyle = MixStyleModule(
            p=self.p,
            alpha=self.alpha,
            eps=self.eps,
            mix=self.mix
        )
        
        # Classification loss
        if self.task_type == 'multi-class':
            self.class_criterion = nn.CrossEntropyLoss()
        else:
            self.class_criterion = nn.BCEWithLogitsLoss()
    
    def extract_features(self, x: torch.Tensor, apply_mixstyle: bool = False) -> torch.Tensor:
        """Extract features, optionally applying MixStyle."""
        features = self.featurizer(x)
        
        # Handle spatial features
        if features.dim() == 4:
            # Apply MixStyle to spatial features before pooling
            if apply_mixstyle and self.training:
                features = self.mixstyle(features)
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif apply_mixstyle and self.training and self.apply_to_features:
            # Apply MixStyle to vector features
            features = apply_mixstyle_to_features(
                features,
                p=self.p,
                alpha=self.alpha,
                eps=self.eps,
                training=self.training
            )
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.extract_features(x, apply_mixstyle=False)
        return self.classifier(features)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update model with MixStyle training.
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B] for multi-class or [B, C] for multi-label
            
        Returns:
            Dictionary with loss values
        """
        # Extract features with MixStyle applied
        features = self.extract_features(x, apply_mixstyle=True)
        
        # Classification
        logits = self.classifier(features)
        loss = self.class_criterion(logits, y)
        
        # Compute accuracy for logging
        with torch.no_grad():
            if self.task_type == 'multi-class':
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
            else:
                pred = (logits > 0).float()
                acc = (pred == y).float().mean()
        
        return {
            'loss': loss,
            'class_loss': loss,
            'accuracy': acc,
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits for input samples.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Class logits [B, num_classes]
        """
        return self.forward(x)
    
    def set_mixstyle_activation(self, status: bool):
        """Enable or disable MixStyle."""
        self.mixstyle.set_activation_status(status)
