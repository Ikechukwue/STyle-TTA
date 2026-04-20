"""
ACVC: Attention Consistency on Visual Corruptions for Single Domain Generalization

Reference:
    Cugu et al., "Attention Consistency on Visual Corruptions for Single-Source 
    Domain Generalization", CVPR Workshops 2022
    https://github.com/ExplainableML/ACVC

ACVC applies visual corruptions (fog, snow, blur, noise, etc.) as data augmentation
and enforces consistency between Class Activation Maps (CAMs) of original and 
corrupted images via KL-divergence.

Key components:
- Visual corruptions: 22 types of image corruptions
- CAM generation: Conv2d(feature_map, classifier_weights)
- Attention Consistency Loss: KL-divergence on CAMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np

from experiments.reference_methods.sdg.algorithms.base import Algorithm


class AttentionConsistencyLoss(nn.Module):
    """
    Attention Consistency Loss between original and corrupted image CAMs.
    
    Uses KL-divergence to enforce consistency on the positive class CAM and
    minimizes attention on top-k negative classes.
    """
    
    def __init__(self, lambd: float = 0.06, temperature: float = 1.0):
        super().__init__()
        self.lambd = lambd
        self.T = temperature
    
    def CAM_neg(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Process CAM for negative class loss.
        Returns negative log-softmax over spatial dimensions.
        """
        B, C, H, W = cam.shape
        result = cam.reshape(B, C, -1)
        result = -F.log_softmax(result / self.T, dim=2) / result.size(2)
        result = result.sum(2)
        return result
    
    def CAM_pos(self, cam: torch.Tensor) -> torch.Tensor:
        """
        Process CAM for positive class loss.
        Returns softmax over spatial dimensions.
        """
        B, C, H, W = cam.shape
        result = cam.reshape(B, C, -1)
        result = F.softmax(result / self.T, dim=2)
        return result
    
    def forward(
        self, 
        cam_orig: torch.Tensor, 
        cam_corrupted_list: list, 
        labels: torch.Tensor,
        k_neg: int = 3
    ) -> torch.Tensor:
        """
        Compute attention consistency loss.
        
        Args:
            cam_orig: CAM of original images [B, C, H, W]
            cam_corrupted_list: List of CAMs from corrupted images
            labels: Ground truth labels [B]
            k_neg: Number of top negative classes to consider
            
        Returns:
            Attention consistency loss
        """
        batch_size = cam_orig.size(0)
        num_classes = cam_orig.size(1)
        device = cam_orig.device
        
        # Process original CAM for negative class selection
        cam_sum = cam_orig.clone().sum(2).sum(2)  # [B, C]
        
        # Mask out true class for top-k selection
        cam_sum_masked = cam_sum.clone()
        cam_sum_masked[range(batch_size), labels] = -float("inf")
        
        # Get top-k negative class indices
        _, topk_neg_indices = torch.topk(cam_sum_masked, k_neg, dim=1)  # [B, k_neg]
        
        # Create mask for negative classes
        neg_mask = torch.zeros(batch_size, num_classes, device=device, dtype=torch.bool)
        neg_mask.scatter_(1, topk_neg_indices, True)
        
        # Negative CAM loss: minimize attention on negative classes
        cam_neg_orig = self.CAM_neg(cam_orig)  # [B, C]
        neg_loss = cam_neg_orig[neg_mask].sum() / batch_size
        
        for cam_corr in cam_corrupted_list:
            cam_neg_corr = self.CAM_neg(cam_corr)
            neg_loss = neg_loss + cam_neg_corr[neg_mask].sum() / batch_size
        neg_loss = neg_loss / (len(cam_corrupted_list) + 1)
        
        # Create mask for positive class
        pos_mask = torch.zeros(batch_size, num_classes, device=device, dtype=torch.bool)
        pos_mask[range(batch_size), labels] = True
        
        # Positive CAM loss: KL-divergence between attention maps
        # Get softmax attention for positive class
        cam_pos_orig = self.CAM_pos(cam_orig)  # [B, C, H*W]
        p_orig = cam_pos_orig[pos_mask]  # [B, H*W]
        
        cam_pos_corrupted = [self.CAM_pos(cam)[pos_mask] for cam in cam_corrupted_list]
        
        # Compute mixture distribution (average of all)
        p_count = 1 + len(cam_pos_corrupted)
        p_mixture = p_orig.detach().clone()
        for p_corr in cam_pos_corrupted:
            p_mixture = p_mixture + p_corr
        p_mixture = torch.clamp(p_mixture / p_count, 1e-7, 1).log()
        
        # KL-divergence loss
        pos_loss = F.kl_div(p_mixture, p_orig, reduction='batchmean')
        for p_corr in cam_pos_corrupted:
            pos_loss = pos_loss + F.kl_div(p_mixture, p_corr, reduction='batchmean')
        pos_loss = pos_loss / p_count
        
        # Total loss
        loss = pos_loss + neg_loss
        return self.lambd * loss


# ============ Visual Corruption Functions ============

def apply_gaussian_noise(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply Gaussian noise corruption."""
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.randn_like(x) * c
    return torch.clamp(x + noise, 0, 1)


def apply_gaussian_blur(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply Gaussian blur corruption."""
    kernel_sizes = [3, 5, 7, 9, 11]
    k = kernel_sizes[severity - 1]
    padding = k // 2
    
    # Create Gaussian kernel
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    
    # Separable convolution
    kernel_h = g.view(1, 1, 1, -1).expand(x.size(1), 1, 1, -1)
    kernel_v = g.view(1, 1, -1, 1).expand(x.size(1), 1, -1, 1)
    
    x = F.conv2d(x, kernel_h, padding=(0, padding), groups=x.size(1))
    x = F.conv2d(x, kernel_v, padding=(padding, 0), groups=x.size(1))
    
    return torch.clamp(x, 0, 1)


def apply_contrast(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply contrast change corruption."""
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    mean = x.mean(dim=[2, 3], keepdim=True)
    return torch.clamp((x - mean) * c + mean, 0, 1)


def apply_brightness(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply brightness change corruption."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    return torch.clamp(x + c, 0, 1)


def apply_saturation(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply saturation change corruption."""
    c = [0.3, 0.5, 0.7, 0.9, 1.0][severity - 1]
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    gray = gray.expand_as(x)
    return torch.clamp(x * c + gray * (1 - c), 0, 1)


def apply_speckle_noise(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply speckle noise corruption."""
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
    noise = torch.randn_like(x) * c
    return torch.clamp(x + x * noise, 0, 1)


def apply_pixelate(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply pixelation corruption."""
    scales = [0.6, 0.5, 0.4, 0.3, 0.25]
    scale = scales[severity - 1]
    
    B, C, H, W = x.shape
    new_h, new_w = int(H * scale), int(W * scale)
    
    # Downsample then upsample
    x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    x = F.interpolate(x, size=(H, W), mode='nearest')
    
    return x


def apply_color_shift(x: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """Apply color channel shift corruption."""
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity - 1]
    shift = torch.rand(x.size(0), 3, 1, 1, device=x.device) * 2 * c - c
    return torch.clamp(x + shift, 0, 1)


# List of available corruptions
CORRUPTIONS = [
    apply_gaussian_noise,
    apply_gaussian_blur,
    apply_contrast,
    apply_brightness,
    apply_saturation,
    apply_speckle_noise,
    apply_pixelate,
    apply_color_shift,
]


def apply_random_corruption(x: torch.Tensor, severity: int = None) -> torch.Tensor:
    """Apply a random corruption to the input."""
    if severity is None:
        severity = np.random.randint(1, 6)
    
    corruption_fn = np.random.choice(CORRUPTIONS)
    return corruption_fn(x, severity)


class ACVC(Algorithm):
    """
    ACVC: Attention Consistency on Visual Corruptions.
    
    Applies visual corruptions as data augmentation and enforces consistency
    between Class Activation Maps (CAMs) of original and corrupted images.
    
    Hyperparameters:
        lambd: Weight for attention consistency loss (default: 0.06)
        temperature: Temperature for softmax in CAM (default: 1.0)
        k_neg: Number of top negative classes to minimize (default: 3)
        corruption_prob: Probability of applying corruption (default: 0.5)
        num_corruptions: Number of corruptions per image (default: 1)
    """
    
    def __init__(
        self,
        featurizer: nn.Module,
        classifier: nn.Module,
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
        
        # Hyperparameters
        self.lambd = hparams.get('lambd', 0.06)
        self.temperature = hparams.get('temperature', 1.0)
        self.k_neg = hparams.get('k_neg', 3)
        self.corruption_prob = hparams.get('corruption_prob', 0.5)
        self.num_corruptions = hparams.get('num_corruptions', 1)
        
        # Attention consistency loss
        self.attention_consistency_loss = AttentionConsistencyLoss(
            lambd=self.lambd,
            temperature=self.temperature
        )
        
        # Feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # Step counter
        self.step_count = 0
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from featurizer."""
        if hasattr(self.featurizer, 'n_outputs'):
            return self.featurizer.n_outputs
        if hasattr(self.featurizer, 'out_features'):
            return self.featurizer.out_features
        if hasattr(self.featurizer, 'num_features'):
            return self.featurizer.num_features
        return 512
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize images from ImageNet normalization."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize images with ImageNet normalization."""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
    def apply_corruptions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random visual corruptions to images.
        """
        # Denormalize first
        x_denorm = self._denormalize(x).clamp(0, 1)
        
        # Apply random corruption
        x_corrupted = apply_random_corruption(x_denorm)
        
        # Normalize back
        return self._normalize(x_corrupted)
    
    def compute_cam(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Class Activation Map from features.
        
        CAM = conv2d(features, classifier_weights)
        """
        # Get classifier weights
        if hasattr(self.classifier, 'weight'):
            weights = self.classifier.weight  # [num_classes, feature_dim]
        else:
            # If classifier is a sequential, try to get the last linear layer
            for layer in reversed(list(self.classifier.modules())):
                if isinstance(layer, nn.Linear):
                    weights = layer.weight
                    break
            else:
                # Fallback: just return features as CAM
                return features.unsqueeze(1)
        
        # Reshape weights for conv operation
        # weights: [num_classes, feature_dim] -> [num_classes, feature_dim, 1, 1]
        if features.dim() == 2:
            # Features are already pooled, can't compute spatial CAM
            # Return dummy CAM
            B = features.size(0)
            return features.unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        
        # features: [B, C, H, W]
        weights = weights.view(weights.size(0), weights.size(1), 1, 1)
        
        # Compute CAM: [B, num_classes, H, W]
        cam = F.conv2d(features, weights)
        
        # Add bias if present
        if hasattr(self.classifier, 'bias') and self.classifier.bias is not None:
            cam = cam + self.classifier.bias.view(1, -1, 1, 1)
        
        return cam
    
    def forward_with_cam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning logits, features, and CAM.
        """
        features = self.featurizer(x)
        
        # Check if features are spatial or pooled
        if features.dim() == 4:
            # Spatial features - compute CAM before pooling
            cam = self.compute_cam(features)
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
            logits = self.classifier(pooled)
        else:
            # Already pooled - CAM won't be spatial
            logits = self.classifier(features)
            cam = features.unsqueeze(2).unsqueeze(3)
        
        return logits, features, cam
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with ACVC.
        """
        # Forward pass on original images
        logits_orig, features_orig, cam_orig = self.forward_with_cam(x)
        
        # Classification loss on original
        cls_loss = self.compute_loss(logits_orig, y)
        
        # Generate corrupted images and compute their CAMs
        cam_corrupted_list = []
        cls_loss_corrupted = torch.tensor(0.0, device=x.device)
        
        for _ in range(self.num_corruptions):
            x_corrupted = self.apply_corruptions(x)
            logits_corr, _, cam_corr = self.forward_with_cam(x_corrupted)
            cam_corrupted_list.append(cam_corr)
            cls_loss_corrupted = cls_loss_corrupted + self.compute_loss(logits_corr, y)
        
        cls_loss_corrupted = cls_loss_corrupted / self.num_corruptions
        
        # Attention consistency loss
        ac_loss = self.attention_consistency_loss(
            cam_orig, cam_corrupted_list, y, k_neg=min(self.k_neg, self.num_classes - 1)
        )
        
        # Total loss
        total_loss = cls_loss + cls_loss_corrupted + ac_loss
        
        self.step_count += 1
        
        return {
            'loss': total_loss,
            'class_loss': cls_loss,
            'class_loss_corrupted': cls_loss_corrupted,
            'attention_consistency_loss': ac_loss,
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (inference mode)."""
        features = self.featurizer(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.classifier(features)
