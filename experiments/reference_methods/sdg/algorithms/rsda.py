"""
RSDA: Random Search Data Augmentation
Paper: "Addressing Model Vulnerability to Distributional Shifts over Image
        Transformation Sets" (ICCV 2019)
Authors: Volpi and Murino

This implementation adapts RSDA for single-domain generalization by:
1. Applying random transformations from a diverse transformation set
2. Maintaining a pool of worst-case transformations that hurt model performance
3. Training on both original and adversarially transformed data

Key transformations:
- autocontrast, brightness, color, contrast, sharpness
- solarize, grayscale
- R/G/B channel enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import random
import math

from .base import Algorithm


# Transformation operations (differentiable PyTorch implementations)

def autocontrast(x: torch.Tensor, cutoff: float = 0.0) -> torch.Tensor:
    """Apply autocontrast to normalize the image histogram."""
    # Compute per-channel min and max
    B, C, H, W = x.shape
    x_flat = x.view(B, C, -1)
    
    # Get min and max values per channel
    x_min = x_flat.min(dim=2, keepdim=True)[0]
    x_max = x_flat.max(dim=2, keepdim=True)[0]
    
    # Normalize to [0, 1]
    scale = 1.0 / (x_max - x_min + 1e-8)
    x_normalized = (x_flat - x_min) * scale
    
    return x_normalized.view(B, C, H, W).clamp(0, 1)


def adjust_brightness(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust image brightness by factor."""
    return (x * factor).clamp(0, 1)


def adjust_saturation(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust image color saturation."""
    # Convert to grayscale
    weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    gray = (x * weights).sum(dim=1, keepdim=True)
    
    # Blend between grayscale and original
    return (gray + factor * (x - gray)).clamp(0, 1)


def adjust_contrast(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust image contrast."""
    mean = x.mean(dim=(-2, -1), keepdim=True)
    return (mean + factor * (x - mean)).clamp(0, 1)


def adjust_sharpness(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust image sharpness using Gaussian blur."""
    if factor == 1.0:
        return x
    
    # Create a simple blur kernel
    kernel = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=x.dtype, device=x.device) / 16.0
    kernel = kernel.view(1, 1, 3, 3).expand(x.size(1), 1, 3, 3)
    
    # Apply blur
    blurred = F.conv2d(x, kernel, padding=1, groups=x.size(1))
    
    # Blend between blurred and original (factor < 1 = more blur, > 1 = sharper)
    return (blurred + factor * (x - blurred)).clamp(0, 1)


def solarize(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Solarize image by inverting pixels above threshold."""
    inverted = 1.0 - x
    return torch.where(x > threshold, inverted, x)


def grayscale(x: torch.Tensor) -> torch.Tensor:
    """Convert to grayscale."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
    gray = (x * weights).sum(dim=1, keepdim=True)
    return gray.expand(-1, 3, -1, -1)


def channel_enhance(x: torch.Tensor, channel: int, delta: float) -> torch.Tensor:
    """Enhance a specific color channel (0=R, 1=G, 2=B)."""
    x = x.clone()
    x[:, channel] = (x[:, channel] + delta).clamp(0, 1)
    return x


class TransformationSet:
    """
    Set of image transformations with random level sampling.
    Based on the transformation set from the RSDA paper.
    """
    
    TRANSFORMATIONS = [
        'identity',
        'autocontrast',
        'brightness',
        'color',  # saturation
        'contrast',
        'sharpness',
        'solarize',
        'grayscale',
        'r_enhance',
        'g_enhance',
        'b_enhance',
    ]
    
    # Level ranges for each transformation
    LEVEL_RANGES = {
        'identity': (0, 0),
        'autocontrast': (0.0, 0.3),
        'brightness': (0.6, 1.4),
        'color': (0.6, 1.4),
        'contrast': (0.6, 1.4),
        'sharpness': (0.6, 1.4),
        'solarize': (0.5, 1.0),  # threshold
        'grayscale': (0, 0),  # no level
        'r_enhance': (-0.3, 0.3),
        'g_enhance': (-0.3, 0.3),
        'b_enhance': (-0.3, 0.3),
    }
    
    def __init__(self, exclude_identity: bool = True):
        self.transformations = self.TRANSFORMATIONS.copy()
        if exclude_identity:
            self.transformations.remove('identity')
    
    def sample_transformation(self) -> Tuple[str, float]:
        """Sample a random transformation with a random level."""
        transf = random.choice(self.transformations)
        level_range = self.LEVEL_RANGES[transf]
        level = random.uniform(level_range[0], level_range[1])
        return transf, level
    
    def sample_transformation_sequence(self, length: int) -> List[Tuple[str, float]]:
        """Sample a sequence of random transformations."""
        return [self.sample_transformation() for _ in range(length)]
    
    @staticmethod
    def apply_transformation(x: torch.Tensor, transf: str, level: float) -> torch.Tensor:
        """Apply a single transformation to the input."""
        if transf == 'identity':
            return x
        elif transf == 'autocontrast':
            return autocontrast(x, cutoff=level)
        elif transf == 'brightness':
            return adjust_brightness(x, factor=level)
        elif transf == 'color':
            return adjust_saturation(x, factor=level)
        elif transf == 'contrast':
            return adjust_contrast(x, factor=level)
        elif transf == 'sharpness':
            return adjust_sharpness(x, factor=level)
        elif transf == 'solarize':
            return solarize(x, threshold=level)
        elif transf == 'grayscale':
            return grayscale(x)
        elif transf == 'r_enhance':
            return channel_enhance(x, channel=0, delta=level)
        elif transf == 'g_enhance':
            return channel_enhance(x, channel=1, delta=level)
        elif transf == 'b_enhance':
            return channel_enhance(x, channel=2, delta=level)
        else:
            return x
    
    def apply_sequence(
        self,
        x: torch.Tensor,
        sequence: List[Tuple[str, float]]
    ) -> torch.Tensor:
        """Apply a sequence of transformations to the input."""
        for transf, level in sequence:
            x = self.apply_transformation(x, transf, level)
        return x


class TransformationPool:
    """
    Pool of worst-case transformations found during training.
    New transformations are added when they reduce model accuracy.
    """
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.pool: List[Tuple[List[Tuple[str, float]], float]] = []
    
    def add(self, sequence: List[Tuple[str, float]], loss: float):
        """Add a transformation sequence with its associated loss."""
        self.pool.append((sequence, loss))
        
        # Keep only the worst-case transformations (highest loss)
        if len(self.pool) > self.max_size:
            self.pool.sort(key=lambda x: x[1], reverse=True)
            self.pool = self.pool[:self.max_size]
    
    def sample(self) -> Optional[List[Tuple[str, float]]]:
        """Sample a transformation from the pool."""
        if len(self.pool) == 0:
            return None
        
        # Sample proportional to loss (higher loss = more likely)
        losses = [item[1] for item in self.pool]
        total_loss = sum(losses)
        if total_loss == 0:
            return random.choice(self.pool)[0]
        
        probs = [l / total_loss for l in losses]
        idx = random.choices(range(len(self.pool)), weights=probs, k=1)[0]
        return self.pool[idx][0]
    
    def __len__(self):
        return len(self.pool)


class RSDA(Algorithm):
    """
    RSDA: Random Search Data Augmentation
    
    Trains with random transformations and maintains a pool of worst-case
    transformations that are challenging for the model.
    
    Args:
        featurizer: Feature extraction network
        classifier: Classification head
        num_classes: Number of output classes
        task_type: 'multi-class' or 'multi-label'
        hparams: Dictionary with hyperparameters:
            - string_length: Number of transformations to concatenate (default: 3)
            - pool_size: Size of transformation pool (default: 50)
            - pool_prob: Probability of sampling from pool vs random (default: 0.5)
            - search_iters: Number of random search iterations per batch (default: 5)
            - use_pool: Whether to use transformation pool (default: True)
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
        self.string_length = hparams.get('string_length', 3)
        self.pool_size = hparams.get('pool_size', 50)
        self.pool_prob = hparams.get('pool_prob', 0.5)
        self.search_iters = hparams.get('search_iters', 5)
        self.use_pool = hparams.get('use_pool', True)
        
        # Transformation set and pool
        self.transf_set = TransformationSet()
        self.transf_pool = TransformationPool(max_size=self.pool_size)
        
        # Classification loss
        if self.task_type == 'multi-class':
            self.class_criterion = nn.CrossEntropyLoss()
        else:
            self.class_criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.featurizer(x)
        
        # Handle spatial features
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        return self.classifier(features)
    
    def random_search(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_iters: int
    ) -> Tuple[List[Tuple[str, float]], float, torch.Tensor]:
        """
        Perform random search to find worst-case transformations.
        
        Args:
            x: Input images
            y: Labels
            n_iters: Number of search iterations
            
        Returns:
            Best (worst-case) transformation sequence, its loss, and transformed images
        """
        best_sequence = None
        best_loss = -float('inf')
        best_images = x
        
        with torch.no_grad():
            for _ in range(n_iters):
                # Sample random transformation sequence
                sequence = self.transf_set.sample_transformation_sequence(self.string_length)
                
                # Apply transformations
                x_transformed = self.transf_set.apply_sequence(x, sequence)
                
                # Compute loss
                logits = self.forward(x_transformed)
                loss = self.class_criterion(logits, y)
                
                # Keep track of worst case
                if loss.item() > best_loss:
                    best_loss = loss.item()
                    best_sequence = sequence
                    best_images = x_transformed
        
        return best_sequence, best_loss, best_images
    
    def get_transformed_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[str, float]], float]:
        """
        Get a transformed batch using random search or pool sampling.
        
        Args:
            x: Input images
            y: Labels
            
        Returns:
            Transformed images, transformation sequence, and loss
        """
        # Decide whether to sample from pool or do random search
        if self.use_pool and len(self.transf_pool) > 0 and random.random() < self.pool_prob:
            # Sample from pool
            sequence = self.transf_pool.sample()
            x_transformed = self.transf_set.apply_sequence(x, sequence)
            with torch.no_grad():
                logits = self.forward(x_transformed)
                loss = self.class_criterion(logits, y).item()
        else:
            # Random search
            sequence, loss, x_transformed = self.random_search(
                x, y, self.search_iters
            )
            
            # Add to pool if it's challenging
            if self.use_pool and sequence is not None:
                self.transf_pool.add(sequence, loss)
        
        return x_transformed, sequence, loss
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update model with RSDA training.
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B] for multi-class or [B, C] for multi-label
            
        Returns:
            Dictionary with loss values
        """
        # Get transformed batch
        x_transformed, sequence, search_loss = self.get_transformed_batch(x, y)
        
        # Forward pass on original images
        logits_orig = self.forward(x)
        loss_orig = self.class_criterion(logits_orig, y)
        
        # Forward pass on transformed images
        logits_transf = self.forward(x_transformed)
        loss_transf = self.class_criterion(logits_transf, y)
        
        # Combined loss (train on both original and transformed)
        total_loss = (loss_orig + loss_transf) / 2
        
        return {
            'loss': total_loss,
            'loss_orig': loss_orig,
            'loss_transf': loss_transf,
            'search_loss': torch.tensor(search_loss, device=x.device),
            'pool_size': torch.tensor(len(self.transf_pool), device=x.device, dtype=torch.float),
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
