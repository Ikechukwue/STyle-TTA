"""
StyDeSty: Min-Max Stylization and Destylization for Single Domain Generalization
Paper: "StyDeSty: Min-Max Stylization and Destylization for Single Domain Generalization" (ICML 2024)
Authors: Songhua Liu, Xin Jin, Xingyi Yang, Jingwen Ye, Xinchao Wang

Key idea: Min-max adversarial training between:
1. Stylization module (AugNet): Generates novel stylized samples
2. Destylization module: Transfers samples to a latent domain using InstanceNorm (AdaIN)

The stylization and destylization modules work adversarially:
- Backbone minimizes: feat_idt + likelihood + cls_loss (destylization)
- AugNet maximizes: feat_idt + cls_loss, minimizes: semantic (stylization)

Reference implementation:
https://github.com/Huage001/StyDeSty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math

from .base import Algorithm


def likelihood(mu: torch.Tensor, log_var: torch.Tensor, tgt_embed: torch.Tensor) -> torch.Tensor:
    """
    Variational alignment loss.
    Encourages the variational distribution to match the target embedding.
    
    Args:
        mu: Mean of variational distribution [B, D]
        log_var: Log variance of variational distribution [B, D]
        tgt_embed: Target embedding [B, D]
        
    Returns:
        Likelihood loss scalar
    """
    return ((mu - tgt_embed) ** 2 / log_var.exp() - log_var).mean()


def club(mu: torch.Tensor, log_var: torch.Tensor, tgt_embed: torch.Tensor) -> torch.Tensor:
    """
    Contrastive Log-ratio Upper Bound (CLUB) for mutual information estimation.
    Used to encourage diversity between augmented and source embeddings.
    
    Args:
        mu: Mean of variational distribution [B, D]
        log_var: Log variance of variational distribution [B, D]
        tgt_embed: Target embedding [B, D]
        
    Returns:
        CLUB loss scalar
    """
    random_idx = torch.randperm(mu.size(0)).long().to(mu.device)
    positive = - (mu - tgt_embed) ** 2 / log_var.exp()
    negative = - (mu - tgt_embed[random_idx]) ** 2 / log_var.exp()
    return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2.


def gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Compute Gaussian kernel matrix between source and target.
    
    Args:
        source: Source samples [B, D]
        target: Target samples [B, D]
        kernel_mul: Kernel multiplier
        kernel_num: Number of kernels
        fix_sigma: Fixed sigma value (None for adaptive)
        
    Returns:
        Kernel matrix [2B, 2B]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + 1e-8)) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with RBF kernel.
    Measures distance between source and target distributions.
    
    Args:
        source: Source samples [B, D]
        target: Target samples [B, D]
        kernel_mul: Kernel multiplier
        kernel_num: Number of kernels
        fix_sigma: Fixed sigma value
        
    Returns:
        MMD loss scalar
    """
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, 
                              kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class ProbMLP(nn.Module):
    """
    Probabilistic MLP with variational reparameterization.
    
    Outputs both predictions and variational parameters (mu, log_var)
    for uncertainty estimation and regularization.
    """
    
    def __init__(self, in_feat: int, out_feat: int, mid_feat: int = 256):
        super().__init__()
        self.log_var = nn.Sequential(nn.Linear(in_feat, mid_feat), nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(in_feat, mid_feat), nn.LeakyReLU())
        self.linear = nn.Linear(mid_feat, out_feat)
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor, factor: float = 0.2) -> torch.Tensor:
        """Reparameterization trick for variational inference."""
        std = log_var.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + factor * std * eps
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predictions and variational parameters.
        
        Returns:
            result: Class predictions [B, num_classes]
            log_var: Log variance [B, mid_feat]
            mu: Mean [B, mid_feat]
            embed: Embedding (mu or reparameterized) [B, mid_feat]
        """
        log_var = self.log_var(x)
        mu = self.mu(x)
        if self.training:
            embed = self.reparameterize(mu, log_var)
        else:
            embed = mu
        result = self.linear(embed)
        return result, log_var, mu, embed


class AugNet(nn.Module):
    """
    Stylization network for generating augmented samples.
    
    Uses learnable spatial and color transformations with InstanceNorm
    to create diverse stylized versions of input images.
    
    During training, random transformations are used.
    During estimation (for backward through AugNet), fixed transformations are used.
    """
    
    def __init__(self, image_size: int = 224, num_transforms: int = 6):
        super().__init__()
        self.image_size = image_size
        
        # Learnable shift parameters for different spatial scales
        # Using smaller kernel sizes for flexibility
        k1, k2, k3, k4 = 9, 13, 17, 5
        s1 = image_size - k1 + 1
        s2 = image_size - k2 + 1
        s3 = image_size - k3 + 1
        s4 = image_size - k4 + 1
        
        self.shift_var1 = nn.Parameter(torch.empty(3, s1, s1))
        nn.init.normal_(self.shift_var1, 1, 0.1)
        self.shift_mean1 = nn.Parameter(torch.zeros(3, s1, s1))
        nn.init.normal_(self.shift_mean1, 0, 0.1)
        
        self.shift_var2 = nn.Parameter(torch.empty(3, s2, s2))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(3, s2, s2))
        nn.init.normal_(self.shift_mean2, 0, 0.1)
        
        self.shift_var3 = nn.Parameter(torch.empty(3, s3, s3))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(3, s3, s3))
        nn.init.normal_(self.shift_mean3, 0, 0.1)
        
        self.shift_var4 = nn.Parameter(torch.empty(3, s4, s4))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(3, s4, s4))
        nn.init.normal_(self.shift_mean4, 0, 0.1)
        
        self.shift_var5 = nn.Parameter(torch.empty(64, 1, 1))
        nn.init.normal_(self.shift_var5, 1, 0.1)
        self.shift_mean5 = nn.Parameter(torch.zeros(64, 1, 1))
        nn.init.normal_(self.shift_mean5, 0, 0.1)
        
        self.norm1 = nn.InstanceNorm2d(3)
        self.norm2 = nn.InstanceNorm2d(64)
        
        # Fixed spatial convolutions for MI estimation
        self.spatial1 = nn.Conv2d(3, 3, k1)
        self.spatial_up1 = nn.ConvTranspose2d(3, 3, k1)
        
        self.spatial2 = nn.Conv2d(3, 3, k2)
        self.spatial_up2 = nn.ConvTranspose2d(3, 3, k2)
        
        self.spatial3 = nn.Conv2d(3, 3, k3)
        self.spatial_up3 = nn.ConvTranspose2d(3, 3, k3)
        
        self.spatial4 = nn.Conv2d(3, 3, k4)
        self.spatial_up4 = nn.ConvTranspose2d(3, 3, k4)
        
        self.spatial5 = nn.Conv2d(3, 64, k4)
        self.spatial_up5 = nn.ConvTranspose2d(64, 3, k4)
        
        self.color = nn.Conv2d(3, 3, 1)
        
        # Freeze fixed parameters
        for param in list(
            list(self.color.parameters()) +
            list(self.spatial1.parameters()) + list(self.spatial_up1.parameters()) +
            list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
            list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
            list(self.spatial4.parameters()) + list(self.spatial_up4.parameters()) +
            list(self.spatial5.parameters()) + list(self.spatial_up5.parameters())
        ):
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, estimation: bool = False) -> torch.Tensor:
        """
        Generate stylized version of input.
        
        Args:
            x: Input images [B, 3, H, W]
            estimation: If True, use fixed transforms for gradient computation
            
        Returns:
            Stylized images [B, 3, H, W]
        """
        device = x.device
        
        if not estimation:
            # Random transforms for diversity
            spatial1 = nn.Conv2d(3, 3, 9).to(device)
            spatial_up1 = nn.ConvTranspose2d(3, 3, 9).to(device)
            spatial2 = nn.Conv2d(3, 3, 13).to(device)
            spatial_up2 = nn.ConvTranspose2d(3, 3, 13).to(device)
            spatial3 = nn.Conv2d(3, 3, 17).to(device)
            spatial_up3 = nn.ConvTranspose2d(3, 3, 17).to(device)
            spatial4 = nn.Conv2d(3, 3, 5).to(device)
            spatial_up4 = nn.ConvTranspose2d(3, 3, 5).to(device)
            spatial5 = nn.Conv2d(3, 64, 5).to(device)
            spatial_up5 = nn.ConvTranspose2d(64, 3, 5).to(device)
            color = nn.Conv2d(3, 3, 1).to(device)
            weight = torch.randn(6).to(device)
            x_c = torch.tanh(F.dropout(color(x), p=0.2))
        else:
            # Fixed transforms for estimation
            spatial1 = self.spatial1
            spatial_up1 = self.spatial_up1
            spatial2 = self.spatial2
            spatial_up2 = self.spatial_up2
            spatial3 = self.spatial3
            spatial_up3 = self.spatial_up3
            spatial4 = self.spatial4
            spatial_up4 = self.spatial_up4
            spatial5 = self.spatial5
            spatial_up5 = self.spatial_up5
            color = self.color
            weight = torch.ones(6).to(device)
            x_c = torch.tanh(color(x))
        
        # Multi-scale spatial transforms with learnable affine
        x_s1down = spatial1(x)
        x_s1down = self.shift_var1 * self.norm1(x_s1down) + self.shift_mean1
        x_s = torch.tanh(spatial_up1(x_s1down))
        
        x_s2down = spatial2(x)
        x_s2down = self.shift_var2 * self.norm1(x_s2down) + self.shift_mean2
        x_s2 = torch.tanh(spatial_up2(x_s2down))
        
        x_s3down = spatial3(x)
        x_s3down = self.shift_var3 * self.norm1(x_s3down) + self.shift_mean3
        x_s3 = torch.tanh(spatial_up3(x_s3down))
        
        x_s4down = spatial4(x)
        x_s4down = self.shift_var4 * self.norm1(x_s4down) + self.shift_mean4
        x_s4 = torch.tanh(spatial_up4(x_s4down))
        
        x_s5down = spatial5(x)
        x_s5down = self.shift_var5 * self.norm2(x_s5down) + self.shift_mean5
        x_s5 = torch.tanh(spatial_up5(x_s5down))
        
        # Weighted combination
        output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2 + 
                  weight[3] * x_s3 + weight[4] * x_s4 + weight[5] * x_s5) / (weight.abs().sum() + 1e-8)
        
        return output


class AugNetSmall(nn.Module):
    """
    Smaller AugNet for low-resolution images (e.g., 32x32).
    """
    
    def __init__(self, image_size: int = 32):
        super().__init__()
        self.image_size = image_size
        
        # Smaller kernel sizes for small images
        k1, k2 = 3, 5
        s1 = image_size - k1 + 1
        s2 = image_size - k2 + 1
        
        self.shift_var = nn.Parameter(torch.empty(3, s1, s1))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(3, s1, s1))
        nn.init.normal_(self.shift_mean, 0, 0.01)
        
        self.shift_var_1 = nn.Parameter(torch.empty(3, s2, s2))
        nn.init.normal_(self.shift_var_1, 1, 0.1)
        self.shift_mean_1 = nn.Parameter(torch.zeros(3, s2, s2))
        nn.init.normal_(self.shift_mean_1, 0, 0.01)
        
        self.norm = nn.InstanceNorm2d(3)
        
        # Fixed spatial convolutions
        self.spatial = nn.Conv2d(3, 3, k1)
        self.spatial_up = nn.ConvTranspose2d(3, 3, k1)
        self.spatial_1 = nn.Conv2d(3, 3, k2)
        self.spatial_up_1 = nn.ConvTranspose2d(3, 3, k2)
        self.color = nn.Conv2d(3, 3, 1)
        
        # Freeze fixed parameters
        for param in list(
            list(self.color.parameters()) +
            list(self.spatial.parameters()) + list(self.spatial_up.parameters()) +
            list(self.spatial_1.parameters()) + list(self.spatial_up_1.parameters())
        ):
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, estimation: bool = False) -> torch.Tensor:
        device = x.device
        
        if not estimation:
            spatial = nn.Conv2d(3, 3, 3).to(device)
            spatial_up = nn.ConvTranspose2d(3, 3, 3).to(device)
            spatial_1 = nn.Conv2d(3, 3, 5).to(device)
            spatial_up_1 = nn.ConvTranspose2d(3, 3, 5).to(device)
            color = nn.Conv2d(3, 3, 1).to(device)
            weight = torch.randn(3).to(device)
            x_c = torch.tanh(F.dropout(color(x), p=0.5))
        else:
            spatial = self.spatial
            spatial_up = self.spatial_up
            spatial_1 = self.spatial_1
            spatial_up_1 = self.spatial_up_1
            color = self.color
            weight = torch.ones(3).to(device)
            x_c = torch.tanh(color(x))
        
        x_down = spatial(x)
        x_down = self.shift_var * self.norm(x_down) + self.shift_mean
        x_s = torch.tanh(spatial_up(x_down))
        
        x_down_1 = spatial_1(x)
        x_down_1 = self.shift_var_1 * self.norm(x_down_1) + self.shift_mean_1
        x_s_1 = torch.tanh(spatial_up_1(x_down_1))
        
        output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s_1) / (weight.abs().sum() + 1e-8)
        return output


class DestylizationLayer(nn.Module):
    """
    Destylization layer using InstanceNorm with learnable affine parameters.
    Placed within the backbone to transfer features to a style-invariant latent domain.
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.instance_norm(x)


class StyDeSty(Algorithm):
    """
    StyDeSty: Min-Max Stylization and Destylization for Single Domain Generalization.
    
    Adversarial training between:
    - Stylization (AugNet): Generates diverse stylized samples
    - Destylization (backbone with InstanceNorm): Learns style-invariant features
    
    The backbone is trained to destylize (minimize distance between original and augmented),
    while AugNet is trained to stylize (maximize that distance while maintaining semantics).
    
    Hyperparameters:
        alpha_feat_idt: Weight for feature identity loss (backbone) (default: 1.0)
        alpha_likelihood: Weight for likelihood loss (backbone) (default: 1.0)
        beta_semantic: Weight for semantic (MMD) loss (AugNet) (default: 1.0)
        beta_likelihood: Weight for CLUB likelihood loss (AugNet) (default: 0.1)
        beta_feat_idt: Weight for feature identity loss (AugNet) (default: 1.0)
        lr_aug: Learning rate for AugNet (default: 0.005)
        aug_weight: Weight for mixing augmented images (default: 0.6)
        mid_feat: Dimension of ProbMLP hidden layer (default: 256)
        small_augnet: Use smaller AugNet for small images (default: False)
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
        
        hparams = hparams or {}
        
        # Loss weights for backbone (destylization)
        self.alpha_feat_idt = hparams.get('alpha_feat_idt', 1.0)
        self.alpha_likelihood = hparams.get('alpha_likelihood', 1.0)
        
        # Loss weights for AugNet (stylization)
        self.beta_semantic = hparams.get('beta_semantic', 1.0)
        self.beta_likelihood = hparams.get('beta_likelihood', 0.1)
        self.beta_feat_idt = hparams.get('beta_feat_idt', 1.0)
        
        # AugNet parameters
        self.lr_aug = hparams.get('lr_aug', 0.005)
        self.aug_weight = hparams.get('aug_weight', 0.6)
        self.use_small_augnet = hparams.get('small_augnet', False)
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        mid_feat = hparams.get('mid_feat', 256)
        
        # Replace classifier with ProbMLP for variational inference
        self.prob_mlp = ProbMLP(self.feature_dim, num_classes, mid_feat)
        
        # Stylization network
        if self.use_small_augnet:
            self.augnet = AugNetSmall(image_size=32)
        else:
            self.augnet = AugNet(image_size=224)
        
        # Optimizer for AugNet (separate from backbone optimizer)
        self.augnet_optimizer = torch.optim.SGD(self.augnet.parameters(), lr=self.lr_aug)
        
        # ImageNet normalization (applied after augmentation)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Step counter
        self.step = 0
    
    def _get_feature_dim(self) -> int:
        """Determine feature dimension from featurizer."""
        if hasattr(self.featurizer, 'n_outputs'):
            return self.featurizer.n_outputs
        if hasattr(self.featurizer, 'out_features'):
            return self.featurizer.out_features
        # Try to infer from last layer
        for module in reversed(list(self.featurizer.modules())):
            if hasattr(module, 'out_features'):
                return module.out_features
            if hasattr(module, 'out_channels'):
                return module.out_channels
        return 512  # Default fallback
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features and handle spatial dimensions."""
        features = self.featurizer(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        features = self.extract_features(x)
        logits, _, _, _ = self.prob_mlp(features)
        return logits
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Update with min-max stylization/destylization training.
        
        Training consists of:
        1. Backbone update (destylization): Minimize distance between augmented and source features
        2. AugNet update (stylization): Maximize that distance while maintaining semantics
        
        Args:
            x: Input images [B, C, H, W]
            y: Labels [B]
            
        Returns:
            Dictionary with loss values
        """
        device = x.device
        B = x.size(0)
        
        losses = {}
        
        # ========== Backbone Update (Destylization) ==========
        # Generate augmented images
        aug_output = self.augnet(x)
        image_aug = torch.sigmoid(aug_output) * self.aug_weight + x * (1. - self.aug_weight)
        
        # Concatenate augmented and source images
        images = torch.cat([image_aug, x], dim=0)
        labels = torch.cat([y, y], dim=0)
        
        # Forward through backbone
        feat = self.extract_features(images)
        pred, log_var, mu, embed = self.prob_mlp(feat)
        
        # Feature identity loss: minimize distance between augmented and source features
        feat_aug = feat[:B]
        feat_src = feat[B:]
        losses['feat_idt'] = F.mse_loss(feat_aug, feat_src) * self.alpha_feat_idt
        
        # Likelihood loss: align source distribution with augmented embeddings
        losses['likelihood'] = likelihood(mu[B:], log_var[B:], embed[:B]) * self.alpha_likelihood
        
        # Classification loss
        if self.task_type == 'multi-class':
            losses['cls'] = F.cross_entropy(pred, labels)
        else:
            losses['cls'] = F.binary_cross_entropy_with_logits(pred, labels.float())
        
        # Total backbone loss
        backbone_loss = losses['cls'] + losses['feat_idt'] + losses['likelihood']
        losses['backbone_loss'] = backbone_loss
        
        # ========== AugNet Update (Stylization) ==========
        self.augnet_optimizer.zero_grad()
        
        # Generate augmented images with estimation mode (for gradient flow)
        aug_output_est = self.augnet(x, estimation=True)
        image_aug_est = torch.sigmoid(aug_output_est) * self.aug_weight + x * (1. - self.aug_weight)
        
        images_est = torch.cat([image_aug_est, x], dim=0)
        
        with torch.no_grad():
            # Detach backbone for AugNet update
            feat_est = self.extract_features(images_est)
        
        # Re-compute with gradients through AugNet
        feat_aug_grad = self.extract_features(image_aug_est)
        
        # Forward through ProbMLP
        pred_est, log_var_est, mu_est, embed_est = self.prob_mlp(feat_est)
        
        # Adversarial feature identity: maximize distance
        losses['adv_feat_idt'] = F.mse_loss(feat_est[:B], feat_est[B:]) * self.beta_feat_idt
        
        # CLUB loss: maximize mutual information upper bound
        losses['adv_likelihood'] = club(mu_est[B:], log_var_est[B:], embed_est[:B]) * self.beta_likelihood
        
        # Classification loss for AugNet (we want good classification)
        if self.task_type == 'multi-class':
            losses['adv_cls'] = F.cross_entropy(pred_est, labels)
        else:
            losses['adv_cls'] = F.binary_cross_entropy_with_logits(pred_est, labels.float())
        
        # Semantic loss: MMD between augmented and source embeddings
        losses['semantic'] = mmd_rbf(embed_est[:B], embed_est[B:]) * self.beta_semantic
        
        # AugNet loss: maximize diversity while maintaining semantics
        # Minimize: semantic + adv_likelihood
        # Maximize: feat_idt + cls
        augnet_loss = losses['semantic'] + losses['adv_likelihood'] - losses['adv_feat_idt'] - losses['adv_cls']
        losses['augnet_loss'] = augnet_loss
        
        # Update AugNet
        augnet_loss.backward()
        self.augnet_optimizer.step()
        
        self.step += 1
        
        # Compute accuracy for logging
        with torch.no_grad():
            if self.task_type == 'multi-class':
                pred_class = pred.argmax(dim=1)
                acc = (pred_class == labels).float().mean()
            else:
                pred_class = (pred > 0).float()
                acc = (pred_class == labels).float().mean()
        
        losses['accuracy'] = acc
        losses['loss'] = backbone_loss  # Main loss for external optimizer
        
        return losses
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class logits.
        
        At inference, only the destylization path is used (backbone + ProbMLP).
        """
        return self.forward(x)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self.augnet.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
