"""
CCSA: Contrastive Cross-domain Semantic Alignment
Paper: "Unified Deep Supervised Domain Adaptation and Generalization" (ICCV 2017)
Authors: Motiian et al.

This implementation adapts CCSA for single-domain generalization by:
1. Using data augmentation to create "target-like" samples from source domain
2. Applying contrastive loss to align original and augmented feature representations
3. Combining classification loss with contrastive semantic alignment loss

Key idea: Learn domain-invariant features by pulling same-class pairs together
and pushing different-class pairs apart in the embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import random

from .base import Algorithm


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for semantic alignment.
    
    For positive pairs (same class): minimize distance
    For negative pairs (different class): maximize distance up to margin
    
    L = y * d² + (1-y) * max(margin - d, 0)²
    
    where y=1 for same class, y=0 for different class
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels1: torch.Tensor,
        labels2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss for pairs.
        
        Args:
            embeddings1: Features from source/original samples [N, D]
            embeddings2: Features from target/augmented samples [M, D]
            labels1: Labels for source samples [N]
            labels2: Labels for target samples [M]
            
        Returns:
            Contrastive loss value
        """
        # Compute pairwise distances
        # embeddings1: [N, D], embeddings2: [M, D]
        # distance: [N, M]
        diff = embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0)  # [N, M, D]
        distances = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-8)  # [N, M]
        
        # Create pair labels: 1 if same class, 0 if different
        pair_labels = (labels1.unsqueeze(1) == labels2.unsqueeze(0)).float()  # [N, M]
        
        # Contrastive loss
        # Positive pairs: minimize distance
        positive_loss = pair_labels * distances ** 2
        
        # Negative pairs: maximize distance up to margin
        negative_loss = (1 - pair_labels) * F.relu(self.margin - distances) ** 2
        
        # Total loss (mean over all pairs)
        loss = (positive_loss + negative_loss).mean()
        
        return loss


class SemanticAlignmentLoss(nn.Module):
    """
    Combined semantic alignment loss for CCSA.
    
    Combines:
    1. Classification loss on source samples
    2. Classification loss on augmented samples (optional)
    3. Contrastive loss for cross-domain alignment
    """
    
    def __init__(self, margin: float = 1.0, alpha: float = 0.25):
        """
        Args:
            margin: Margin for contrastive loss
            alpha: Weight for contrastive loss vs classification loss
                   Loss = (1-alpha)*class_loss + alpha*contrastive_loss
        """
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        self.alpha = alpha
    
    def forward(
        self,
        features_source: torch.Tensor,
        features_aug: torch.Tensor,
        labels_source: torch.Tensor,
        labels_aug: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute semantic alignment loss.
        
        Args:
            features_source: Features from original samples [N, D]
            features_aug: Features from augmented samples [N, D]
            labels_source: Labels for source samples [N]
            labels_aug: Labels for augmented samples [N]
            
        Returns:
            Contrastive semantic alignment loss
        """
        return self.contrastive_loss(
            features_source, features_aug, labels_source, labels_aug
        )


def color_jitter(x: torch.Tensor, brightness: float = 0.4, contrast: float = 0.4,
                 saturation: float = 0.4, hue: float = 0.1) -> torch.Tensor:
    """Apply random color jittering."""
    # Simple brightness adjustment
    if random.random() > 0.5:
        factor = 1.0 + random.uniform(-brightness, brightness)
        x = x * factor
    
    # Simple contrast adjustment
    if random.random() > 0.5:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * factor + mean
    
    return x.clamp(0, 1)


def random_grayscale(x: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """Randomly convert to grayscale."""
    if random.random() < p:
        # Convert to grayscale using luminance weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device)
        gray = (x * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
        x = gray.expand(-1, 3, -1, -1)
    return x


def gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Add Gaussian noise."""
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(0, 1)


def random_erasing(x: torch.Tensor, p: float = 0.5, scale: tuple = (0.02, 0.33),
                   ratio: tuple = (0.3, 3.3)) -> torch.Tensor:
    """Random erasing augmentation."""
    if random.random() > p:
        return x
    
    B, C, H, W = x.shape
    area = H * W
    
    for i in range(B):
        for _ in range(10):  # Max attempts
            target_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])
            
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
            
            if h < H and w < W:
                y = random.randint(0, H - h)
                x_pos = random.randint(0, W - w)
                x[i, :, y:y+h, x_pos:x_pos+w] = torch.rand(C, h, w, device=x.device)
                break
    
    return x


class CCSA(Algorithm):
    """
    CCSA: Contrastive Cross-domain Semantic Alignment
    
    Adapts CCSA for single-domain generalization by using augmentations
    to simulate domain shift and learning domain-invariant features through
    contrastive semantic alignment.
    
    Key components:
    1. Feature extractor (shared for source and augmented samples)
    2. Classifier for semantic classification
    3. Contrastive loss for aligning representations across domains
    
    Args:
        featurizer: Feature extraction network
        classifier: Classification head
        num_classes: Number of output classes
        task_type: 'multi-class' or 'multi-label'
        hparams: Dictionary with hyperparameters:
            - margin: Margin for contrastive loss (default: 1.0)
            - alpha: Weight for contrastive loss (default: 0.25)
            - aug_strength: Strength of augmentations (default: 0.5)
            - use_color_jitter: Use color jitter augmentation (default: True)
            - use_grayscale: Use random grayscale (default: True)
            - use_noise: Use Gaussian noise (default: True)
            - use_erasing: Use random erasing (default: True)
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
        self.margin = hparams.get('margin', 1.0)
        self.alpha = hparams.get('alpha', 0.25)
        self.aug_strength = hparams.get('aug_strength', 0.5)
        self.use_color_jitter = hparams.get('use_color_jitter', True)
        self.use_grayscale = hparams.get('use_grayscale', True)
        self.use_noise = hparams.get('use_noise', True)
        self.use_erasing = hparams.get('use_erasing', True)
        
        # Semantic alignment loss
        self.csa_loss = SemanticAlignmentLoss(margin=self.margin, alpha=self.alpha)
        
        # Classification loss
        if self.task_type == 'multi-class':
            self.class_criterion = nn.CrossEntropyLoss()
        else:
            self.class_criterion = nn.BCEWithLogitsLoss()
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply domain-shift augmentations to create "target-like" samples.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Augmented images [B, C, H, W]
        """
        x_aug = x.clone()
        
        if self.use_color_jitter:
            x_aug = color_jitter(
                x_aug,
                brightness=self.aug_strength,
                contrast=self.aug_strength,
                saturation=self.aug_strength,
                hue=self.aug_strength * 0.25
            )
        
        if self.use_grayscale:
            x_aug = random_grayscale(x_aug, p=0.2)
        
        if self.use_noise:
            if random.random() > 0.5:
                x_aug = gaussian_noise(x_aug, std=self.aug_strength * 0.2)
        
        if self.use_erasing:
            x_aug = random_erasing(x_aug, p=0.3)
        
        return x_aug
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.featurizer(x)
        
        # Handle spatial features (from CNN without global pooling)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return self.classifier(features)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update model with CCSA training.
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B] for multi-class or [B, C] for multi-label
            
        Returns:
            Dictionary with loss values
        """
        batch_size = x.size(0)
        
        # Create augmented samples (simulating target domain)
        x_aug = self.augment(x)
        
        # Extract features for both source and augmented samples
        features_source = self.featurizer(x)
        features_aug = self.featurizer(x_aug)
        
        # Handle spatial features
        if features_source.dim() == 4:
            features_source_pooled = F.adaptive_avg_pool2d(features_source, 1).flatten(1)
            features_aug_pooled = F.adaptive_avg_pool2d(features_aug, 1).flatten(1)
        else:
            features_source_pooled = features_source
            features_aug_pooled = features_aug
        
        # Classification on source samples
        logits_source = self.classifier(features_source_pooled)
        class_loss_source = self.class_criterion(logits_source, y)
        
        # Classification on augmented samples
        logits_aug = self.classifier(features_aug_pooled)
        class_loss_aug = self.class_criterion(logits_aug, y)
        
        # Combined classification loss
        class_loss = (class_loss_source + class_loss_aug) / 2
        
        # Contrastive semantic alignment loss
        # Normalize features for contrastive loss
        features_source_norm = F.normalize(features_source_pooled, dim=1)
        features_aug_norm = F.normalize(features_aug_pooled, dim=1)
        
        csa_loss = self.csa_loss(
            features_source_norm,
            features_aug_norm,
            y,
            y  # Same labels since augmented samples have same class
        )
        
        # Combined loss: (1-alpha)*class_loss + alpha*csa_loss
        total_loss = (1 - self.alpha) * class_loss + self.alpha * csa_loss
        
        return {
            'loss': total_loss,
            'class_loss': class_loss,
            'class_loss_source': class_loss_source,
            'class_loss_aug': class_loss_aug,
            'csa_loss': csa_loss
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
