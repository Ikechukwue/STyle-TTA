"""
PDEN: Progressive Domain Expansion Network
Paper: "Progressive Domain Expansion Network for Single Domain Generalization"
Authors: Li et al.

This implementation adapts PDEN for single-domain generalization by:
1. Using a domain generator network with AdaIN to synthesize domain-shifted images
2. Training classifier on both original and generated domains
3. Using contrastive learning to align representations across domains
4. Adversarial training to maximize domain diversity while maintaining class info

Key components:
- AdaIN-based domain generator (injects random style via adaptive instance normalization)
- SupConLoss for contrastive representation learning
- Diversity loss to encourage varied domain generations
- Cycle consistency loss (optional) to preserve content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import random

from .base import Algorithm


class AdaIN2d(nn.Module):
    """
    Adaptive Instance Normalization layer.
    Injects style information via learned affine transformation.
    """
    
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
    
    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class DomainGenerator(nn.Module):
    """
    CNN-based domain generator with AdaIN for style injection.
    Takes source images and generates domain-shifted versions.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 32,
        zdim: int = 10,
        kernel_size: int = 3
    ):
        super().__init__()
        self.zdim = zdim
        stride = (kernel_size - 1) // 2
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size, 1, stride)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size, 1, stride)
        
        # AdaIN for style injection
        self.adain = AdaIN2d(zdim, hidden_dim * 2)
        
        # Decoder
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size, 1, stride)
        self.conv4 = nn.Conv2d(hidden_dim * 4, in_channels, kernel_size, 1, stride)
    
    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        rand: bool = True
    ) -> torch.Tensor:
        """
        Generate domain-shifted images.
        
        Args:
            x: Input images [B, C, H, W]
            z: Style vector [B, zdim] (optional, will be sampled if None and rand=True)
            rand: Whether to sample random style
            
        Returns:
            Domain-shifted images [B, C, H, W]
        """
        # Encode
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        
        # Apply AdaIN with style
        if rand:
            z = torch.randn(x.size(0), self.zdim, device=x.device)
        if z is not None:
            h = self.adain(h, z)
        
        # Decode
        h = F.relu(self.conv3(h))
        out = torch.sigmoid(self.conv4(h))
        
        return out


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss.
    From: "Supervised Contrastive Learning" (Khosla et al., 2020)
    
    Supports both standard contrastive learning and adversarial mode
    for domain expansion.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        adv: bool = False
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature tensor [B, n_views, D]
            labels: Class labels [B]
            adv: If True, use adversarial mode (maximize distance to same class)
            
        Returns:
            Contrastive loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        if len(features.shape) < 3:
            raise ValueError('Features needs to be [B, n_views, D]')
        
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        # Create mask based on labels
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        
        if adv:
            # Adversarial mode: maximize distance to same class
            log_prob = torch.log(1 - exp_logits / (exp_logits.sum(1, keepdim=True) + 1e-6) + 1e-6)
        else:
            # Standard mode: minimize distance to same class
            log_prob = torch.log(exp_logits / (exp_logits.sum(1, keepdim=True) + 1e-6) + 1e-6)
        
        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        # Loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class PDEN(Algorithm):
    """
    PDEN: Progressive Domain Expansion Network
    
    Trains a domain generator to synthesize diverse domain-shifted images
    and uses contrastive learning to align representations across domains.
    
    Args:
        featurizer: Feature extraction network
        classifier: Classification head
        num_classes: Number of output classes
        task_type: 'multi-class' or 'multi-label'
        hparams: Dictionary with hyperparameters:
            - zdim: Dimension of style vector (default: 10)
            - gen_hidden: Hidden dimension of generator (default: 32)
            - w_cls: Weight for classification loss on generated (default: 1.0)
            - w_con: Weight for contrastive loss (default: 1.0)
            - w_div: Weight for diversity loss (default: 1.0)
            - w_cyc: Weight for cycle consistency loss (default: 10.0)
            - div_thresh: Threshold for diversity loss (default: 0.1)
            - temperature: Temperature for contrastive loss (default: 0.07)
            - lr_gen: Learning rate for generator (default: 1e-3)
            - use_cycle: Whether to use cycle consistency (default: True)
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
        self.zdim = hparams.get('zdim', 10)
        self.gen_hidden = hparams.get('gen_hidden', 32)
        self.w_cls = hparams.get('w_cls', 1.0)
        self.w_con = hparams.get('w_con', 1.0)
        self.w_div = hparams.get('w_div', 1.0)
        self.w_cyc = hparams.get('w_cyc', 10.0)
        self.div_thresh = hparams.get('div_thresh', 0.1)
        self.temperature = hparams.get('temperature', 0.07)
        self.lr_gen = hparams.get('lr_gen', 1e-3)
        self.use_cycle = hparams.get('use_cycle', True)
        
        # Domain generator
        self.generator = DomainGenerator(
            in_channels=3,
            hidden_dim=self.gen_hidden,
            zdim=self.zdim
        )
        
        # Reconstruction network for cycle consistency
        if self.use_cycle:
            self.reconstructor = DomainGenerator(
                in_channels=3,
                hidden_dim=self.gen_hidden,
                zdim=self.zdim
            )
        else:
            self.reconstructor = None
        
        # Contrastive loss
        self.con_criterion = SupConLoss(temperature=self.temperature)
        
        # Classification loss
        if self.task_type == 'multi-class':
            self.class_criterion = nn.CrossEntropyLoss()
        else:
            self.class_criterion = nn.BCEWithLogitsLoss()
        
        # Generator optimizer (separate from main optimizer)
        gen_params = list(self.generator.parameters())
        if self.reconstructor is not None:
            gen_params += list(self.reconstructor.parameters())
        self.gen_optimizer = torch.optim.Adam(gen_params, lr=self.lr_gen)
    
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
        Update model with PDEN training.
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B] for multi-class or [B, C] for multi-label
            
        Returns:
            Dictionary with loss values
        """
        batch_size = x.size(0)
        
        # Generate domain-shifted images (two different random samples for diversity)
        x_gen1 = self.generator(x, rand=True)
        x_gen2 = self.generator(x, rand=True)
        
        # Extract features
        z_src = self.extract_features(x)
        z_gen = self.extract_features(x_gen1)
        
        # Normalize features for contrastive loss
        z_src_norm = F.normalize(z_src, dim=1)
        z_gen_norm = F.normalize(z_gen, dim=1)
        
        # Classification losses
        logits_src = self.classifier(z_src)
        logits_gen = self.classifier(z_gen)
        
        cls_loss_src = self.class_criterion(logits_src, y)
        cls_loss_gen = self.class_criterion(logits_gen, y)
        
        # Contrastive loss (align source and generated features)
        # Stack features: [B, 2, D]
        features = torch.stack([z_gen_norm, z_src_norm], dim=1)
        con_loss = self.con_criterion(features, y, adv=False)
        
        # Update classifier with combined loss
        cls_loss = cls_loss_src + self.w_cls * cls_loss_gen + self.w_con * con_loss
        
        # ============ Update Generator ============
        # Adversarial contrastive loss (maximize domain gap while preserving class)
        z_gen_adv = self.extract_features(x_gen1)
        z_gen_adv_norm = F.normalize(z_gen_adv, dim=1)
        features_adv = torch.stack([z_gen_adv_norm, z_src_norm.detach()], dim=1)
        con_loss_adv = self.con_criterion(features_adv, y, adv=True)
        
        # Diversity loss (encourage different outputs for different random styles)
        div_loss = (x_gen1 - x_gen2).abs().mean(dim=[1, 2, 3])
        div_loss = div_loss.clamp(max=self.div_thresh).mean()
        
        # Cycle consistency loss
        if self.use_cycle and self.reconstructor is not None:
            x_rec = self.reconstructor(x_gen1, rand=False)
            cyc_loss = F.mse_loss(x_rec, x)
        else:
            cyc_loss = torch.tensor(0.0, device=x.device)
        
        # Generator loss: maximize domain shift while preserving class info
        gen_loss = (
            self.w_cls * cls_loss_gen 
            - self.w_div * div_loss  # Negative because we want to maximize diversity
            + self.w_cyc * cyc_loss 
            + self.w_con * con_loss_adv
        )
        
        # Update generator
        self.gen_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        self.gen_optimizer.step()
        
        return {
            'loss': cls_loss,
            'cls_loss_src': cls_loss_src,
            'cls_loss_gen': cls_loss_gen,
            'con_loss': con_loss,
            'con_loss_adv': con_loss_adv,
            'div_loss': div_loss,
            'cyc_loss': cyc_loss,
            'gen_loss': gen_loss,
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
    
    def to(self, device):
        """Move model to device, including generator optimizer."""
        super().to(device)
        # Recreate optimizer with parameters on new device
        gen_params = list(self.generator.parameters())
        if self.reconstructor is not None:
            gen_params += list(self.reconstructor.parameters())
        self.gen_optimizer = torch.optim.Adam(gen_params, lr=self.lr_gen)
        return self
