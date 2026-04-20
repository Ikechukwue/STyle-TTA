"""
AdvST: Adversarial Style Transfer for Single Domain Generalization

Reference:
    Zheng et al., "AdvST: Revisiting Data Augmentations for Single Domain 
    Generalization", AAAI 2024
    https://github.com/gtzheng/AdvST

AdvST uses learnable semantic perturbations (HSV, rotation, translation, contrast,
sharpness, etc.) to generate adversarial augmentations. The augmentation parameters
are optimized to maximize classification loss while maintaining semantic consistency.

Key components:
- SemanticAugment: Learnable augmentation with differentiable kornia operations
- Adversarial optimization: maximize(class_loss - gamma * semantic_loss + eta * entropy)
- Contrastive learning: SupConLoss for consistency between original and augmented
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Callable
import numpy as np
import random

from experiments.reference_methods.sdg.algorithms.base import Algorithm


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of predictions."""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        
        if len(features.shape) < 3:
            features = features.unsqueeze(1)
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


# ============ Differentiable Augmentation Operations ============

def hsv_aug(x: torch.Tensor, hsv: torch.Tensor) -> torch.Tensor:
    """HSV color augmentation."""
    B = x.shape[0]
    # Simple approximation without kornia dependency
    # Shift RGB values based on HSV-like parameters
    shift = hsv.view(B, 3, 1, 1)
    return torch.clamp(x + shift * 0.1, 0, 1)


def rotate_aug(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation augmentation using affine grid."""
    B = x.shape[0]
    angle_rad = torch.clamp(angle, 0.01, 1) * 2 * np.pi  # 0 to 360 degrees
    
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    # Build rotation matrix
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = cos_a.squeeze()
    theta[:, 0, 1] = -sin_a.squeeze()
    theta[:, 1, 0] = sin_a.squeeze()
    theta[:, 1, 1] = cos_a.squeeze()
    
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')


def translate_aug(x: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Translation augmentation."""
    B = x.shape[0]
    trans = torch.clamp(trans, -1, 1) * 0.1  # Max 10% translation
    
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = trans[:, 0] if trans.dim() > 1 else trans
    theta[:, 1, 2] = trans[:, 1] if trans.dim() > 1 and trans.size(1) > 1 else 0
    
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')


def contrast_aug(x: torch.Tensor, con: torch.Tensor) -> torch.Tensor:
    """Contrast augmentation."""
    B = x.shape[0]
    con = torch.clamp(con, 0.5, 1.5).view(B, 1, 1, 1)
    mean = x.mean(dim=[2, 3], keepdim=True)
    return torch.clamp((x - mean) * con + mean, 0, 1)


def brightness_aug(x: torch.Tensor, bright: torch.Tensor) -> torch.Tensor:
    """Brightness augmentation."""
    B = x.shape[0]
    bright = torch.clamp(bright, -0.3, 0.3).view(B, 1, 1, 1)
    return torch.clamp(x + bright, 0, 1)


def scale_aug(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Scale augmentation."""
    B = x.shape[0]
    factor = torch.clamp(factor, 0.8, 1.2).view(B)
    
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1.0 / factor
    theta[:, 1, 1] = 1.0 / factor
    
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')


def shear_aug(x: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
    """Shear augmentation."""
    B = x.shape[0]
    shear = torch.clamp(shear, -0.2, 0.2)
    
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    if shear.dim() > 1 and shear.size(1) >= 2:
        theta[:, 0, 1] = shear[:, 0]
        theta[:, 1, 0] = shear[:, 1]
    else:
        theta[:, 0, 1] = shear.squeeze()
    
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')


class SemanticAugment(nn.Module):
    """
    Learnable semantic augmentation module.
    
    Contains learnable parameters for each augmentation operation.
    """
    
    def __init__(self, batch_size: int, ops: List[Tuple], device: torch.device):
        super().__init__()
        self.ops_funcs = [op[0] for op in ops]
        
        params = []
        for op_func, (min_val, max_val), num_params in ops:
            init_val = torch.rand(batch_size, num_params, device=device) * (max_val - min_val) + min_val
            if num_params == 1:
                init_val = init_val.squeeze(1)
            params.append(nn.Parameter(init_val))
        
        self.params = nn.ParameterList(params)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, op in enumerate(self.ops_funcs):
            x = torch.clamp(op(x, self.params[i]), 0, 1)
        return x


class SemanticPerturbation:
    """
    Manages sampling of semantic augmentation combinations.
    """
    
    # Available augmentation operations: (function, (min, max), num_params)
    semantics_list = [
        (hsv_aug, (-0.5, 0.5), 3),
        (rotate_aug, (0.01, 0.5), 1),
        (translate_aug, (-0.5, 0.5), 2),
        (contrast_aug, (0.5, 1.5), 1),
        (brightness_aug, (-0.3, 0.3), 1),
        (scale_aug, (0.8, 1.2), 1),
        (shear_aug, (-0.2, 0.2), 2),
    ]
    
    def __init__(self, max_ops: int = 3):
        """
        Initialize with combinations of 1 to max_ops operations.
        """
        self.max_ops = max_ops
        num_ops = len(self.semantics_list)
        
        # Generate all combinations of 1 to max_ops operations
        self.op_combinations = []
        for i in range(num_ops):
            self.op_combinations.append([i])
        
        # Add 2-op combinations
        if max_ops >= 2:
            for i in range(num_ops):
                for j in range(num_ops):
                    if i != j:
                        self.op_combinations.append([i, j])
        
        # Add 3-op combinations
        if max_ops >= 3:
            for i in range(num_ops):
                for j in range(num_ops):
                    for k in range(num_ops):
                        if i != j and j != k and i != k:
                            self.op_combinations.append([i, j, k])
        
        # Uniform probability over all combinations
        self.probs = np.ones(len(self.op_combinations)) / len(self.op_combinations)
    
    def sample(self, batch_size: int, device: torch.device) -> SemanticAugment:
        """Sample a random augmentation combination."""
        idx = np.random.choice(len(self.op_combinations), p=self.probs)
        op_indices = self.op_combinations[idx]
        ops = [self.semantics_list[i] for i in op_indices]
        return SemanticAugment(batch_size, ops, device)


class ADVST(Algorithm):
    """
    AdvST: Adversarial Style Transfer for Single Domain Generalization.
    
    Uses learnable semantic perturbations optimized adversarially to create
    diverse augmentations that maintain semantic consistency.
    
    Hyperparameters:
        gamma: Weight for semantic distance regularization (default: 10.0)
        eta: Weight for entropy maximization (default: 10.0)
        eta_min: Weight for entropy regularization during minimization (default: 0.01)
        beta: Weight for contrastive loss (default: 1.0)
        lr_max: Learning rate for augmentation optimization (default: 5.0)
        loops_adv: Number of adversarial optimization steps (default: 50)
        gen_freq: Frequency of generating new augmentations (default: 1)
        max_ops: Maximum number of augmentation operations (default: 3)
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
        self.gamma = hparams.get('gamma', 10.0)
        self.eta = hparams.get('eta', 10.0)
        self.eta_min = hparams.get('eta_min', 0.01)
        self.beta = hparams.get('beta', 1.0)
        self.lr_max = hparams.get('lr_max', 5.0)
        self.loops_adv = hparams.get('loops_adv', 50)
        self.max_ops = hparams.get('max_ops', 3)
        self.use_contrastive = hparams.get('use_contrastive', True)
        
        # Semantic perturbation sampler
        self.semantic_config = SemanticPerturbation(max_ops=self.max_ops)
        
        # Contrastive loss
        self.contrastive_loss = SupConLoss(temperature=0.07)
        
        # Projection head for contrastive learning
        self.feature_dim = self._get_feature_dim()
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 128)
        )
        
        # MSE for semantic distance
        self.dist_fn = nn.MSELoss()
        
        # Step counter
        self.step_count = 0
        
        # Data pool for augmented images
        self.aug_pool = []
        self.pool_size = hparams.get('pool_size', 5)
        
        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
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
    
    def generate_adversarial(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial augmentations via semantic perturbation.
        
        Optimizes augmentation parameters to maximize:
        class_loss - gamma * semantic_dist + eta * entropy
        """
        batch_size = x.size(0)
        device = x.device
        
        # Get original embeddings (frozen)
        with torch.no_grad():
            orig_features = self.featurizer(x)
            orig_embedding = orig_features.detach().clone()
        
        # Sample semantic perturbation
        semantic_perturb = self.semantic_config.sample(batch_size, device)
        
        # Optimizer for augmentation parameters
        optimizer = torch.optim.RMSprop(semantic_perturb.parameters(), lr=self.lr_max)
        
        # Denormalize input for augmentation
        x_denorm = self._denormalize(x).clamp(0, 1)
        
        prev_loss = 0.0
        for _ in range(self.loops_adv):
            # Apply augmentation
            x_aug = semantic_perturb(x_denorm)
            x_aug_norm = self._normalize(x_aug)
            
            # Forward pass
            features = self.featurizer(x_aug_norm)
            logits = self.classifier(features)
            
            # Compute adversarial loss
            cls_loss = self.compute_loss(logits, y)
            semantic_dist = self.dist_fn(features, orig_embedding)
            ent_loss = entropy_loss(logits)
            
            # Maximize: class_loss - gamma * semantic_dist + eta * entropy
            loss = cls_loss - self.gamma * semantic_dist + self.eta * ent_loss
            
            # Gradient ascent (negate loss for maximization)
            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
            
            # Early stopping if converged
            diff = abs((loss - prev_loss).item())
            if diff < 0.1:
                break
            prev_loss = loss.item()
        
        # Generate final augmented images
        with torch.no_grad():
            x_aug = semantic_perturb(x_denorm)
            x_aug_norm = self._normalize(x_aug)
        
        return x_aug_norm
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with AdvST.
        """
        batch_size = x.size(0)
        
        # Generate adversarial augmentations
        if self.training:
            x_aug = self.generate_adversarial(x, y)
            
            # Combine original and augmented for training
            x_combined = torch.cat([x, x_aug], dim=0)
            y_combined = torch.cat([y, y], dim=0)
        else:
            x_combined = x
            y_combined = y
        
        # Forward pass
        features = self.featurizer(x_combined)
        logits = self.classifier(features)
        
        # Classification loss
        cls_loss = self.compute_loss(logits, y_combined)
        
        # Contrastive loss (if enabled)
        con_loss = torch.tensor(0.0, device=x.device)
        if self.use_contrastive and self.training and batch_size > 1:
            # Project features
            proj = self.projection_head(features)
            proj = F.normalize(proj, dim=1)
            
            # Split into original and augmented
            proj_orig = proj[:batch_size].unsqueeze(1)
            proj_aug = proj[batch_size:].unsqueeze(1)
            
            # Stack for contrastive loss
            proj_combined = torch.cat([proj_orig, proj_aug], dim=1)
            con_loss = self.contrastive_loss(proj_combined, y)
        
        # Entropy regularization (minimize entropy for confident predictions)
        ent_reg = entropy_loss(logits)
        
        # Total loss
        total_loss = cls_loss + self.beta * con_loss - self.eta_min * ent_reg
        
        self.step_count += 1
        
        return {
            'loss': total_loss,
            'class_loss': cls_loss,
            'contrastive_loss': con_loss,
            'entropy': ent_reg.detach(),
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions (inference mode)."""
        features = self.featurizer(x)
        return self.classifier(features)
