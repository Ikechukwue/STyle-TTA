"""
L2D: Learning to Diversify for Single Domain Generalization

Reference:
    Wang et al., "Learning to Diversify for Single Domain Generalization",
    ICCV 2021
    https://github.com/BUserName/Learning_to_diversify

L2D uses a learnable augmentation network (AugNet/Convertor) to generate diverse
augmented images. The training alternates between:
1. Training the classifier with contrastive loss and likelihood maximization
2. Training the augmentation network to maximize diversity (conditional MMD)

Key components:
- AugNet: Learnable augmentation network with color and spatial transformations
- Variational encoder: mu and logvar for reparameterization trick
- Contrastive loss: SupConLoss for semantic consistency
- CLUB: Contrastive Log-ratio Upper Bound for MI estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from torchvision import transforms

from experiments.reference_methods.sdg.algorithms.base import Algorithm


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss from Khosla et al.
    https://arxiv.org/abs/2004.11362
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Shape [batch_size, n_views, feature_dim]
            labels: Shape [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Create mask based on labels
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
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
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class AugNet(nn.Module):
    """
    Learnable augmentation network that generates diverse augmented images.
    Uses color transformations and spatial convolutions with learnable shift parameters.
    """
    
    def __init__(self, image_size: int = 224, noise_level: float = 0.1):
        super().__init__()
        
        self.noise_level = nn.Parameter(torch.zeros(1))
        
        # Color transformation
        self.color = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.InstanceNorm2d(3),
            nn.Conv2d(3, 3, 1)
        )
        
        # Spatial transformations with different kernel sizes
        # We use smaller kernels for efficiency
        self.spatial1 = nn.Conv2d(3, 3, 5, padding=2)
        self.spatial2 = nn.Conv2d(3, 3, 7, padding=3)
        self.spatial3 = nn.Conv2d(3, 3, 3, padding=1)
        
        # Learnable shift parameters for normalization
        self.shift_var1 = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.shift_mean1 = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.shift_var2 = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.shift_mean2 = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.shift_var3 = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.shift_mean3 = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        self.norm = nn.InstanceNorm2d(3)
        
        # Learnable weights for combining transformations
        self.weights = nn.Parameter(torch.ones(4) / 4)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, estimation: bool = False) -> torch.Tensor:
        """
        Generate augmented images.
        
        Args:
            x: Input images
            estimation: If True, use learned parameters. If False, use random init.
        """
        # Add noise
        x_noisy = x + torch.randn_like(x) * self.noise_level * 0.01
        
        # Color transformation
        x_color = torch.tanh(self.color(x_noisy))
        
        # Spatial transformations with shift normalization
        x_s1 = self.spatial1(x_noisy)
        x_s1 = self.shift_var1 * self.norm(x_s1) + self.shift_mean1
        x_s1 = torch.tanh(x_s1)
        
        x_s2 = self.spatial2(x_noisy)
        x_s2 = self.shift_var2 * self.norm(x_s2) + self.shift_mean2
        x_s2 = torch.tanh(x_s2)
        
        x_s3 = self.spatial3(x_noisy)
        x_s3 = self.shift_var3 * self.norm(x_s3) + self.shift_mean3
        x_s3 = torch.tanh(x_s3)
        
        # Weighted combination
        weights = F.softmax(self.weights, dim=0)
        output = weights[0] * x_color + weights[1] * x_s1 + weights[2] * x_s2 + weights[3] * x_s3
        
        return output


class VariationalHead(nn.Module):
    """
    Variational head for reparameterization trick.
    Outputs mu and logvar for embedding.
    """
    
    def __init__(self, in_features: int, out_features: int = 512):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(inplace=True)
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, train: bool = True) -> Dict[str, torch.Tensor]:
        mu = self.p_mu(x)
        logvar = self.p_logvar(x)
        
        if train:
            # Reparameterization trick
            std = logvar.div(2).exp()
            eps = torch.randn_like(std)
            embedding = mu + 0.2 * std * eps
        else:
            embedding = mu
        
        return {'mu': mu, 'logvar': logvar, 'embedding': embedding}


def loglikeli(mu: torch.Tensor, logvar: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
    """
    Compute log-likelihood for variational inference.
    """
    return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).mean()


def club(mu: torch.Tensor, logvar: torch.Tensor, y_samples: torch.Tensor) -> torch.Tensor:
    """
    CLUB: Contrastive Log-ratio Upper Bound for mutual information estimation.
    """
    batch_size = y_samples.shape[0]
    random_index = torch.randperm(batch_size).to(y_samples.device)
    
    positive = -(mu - y_samples) ** 2 / (logvar.exp() + 1e-8)
    negative = -(mu - y_samples[random_index]) ** 2 / (logvar.exp() + 1e-8)
    
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    return upper_bound / 2.0


def gaussian_kernel(source: torch.Tensor, target: torch.Tensor, 
                    kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """
    Compute Gaussian kernel for MMD.
    """
    n_samples = source.size(0) + target.size(0)
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples + 1e-8)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (bw + 1e-8)) for bw in bandwidth_list]
    
    return sum(kernel_val)


def mmd_rbf(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute MMD with RBF kernel.
    """
    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def conditional_mmd_rbf(source: torch.Tensor, target: torch.Tensor, 
                        labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute conditional MMD (class-wise MMD).
    """
    loss = 0.0
    count = 0
    
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 1:
            source_i = source[mask]
            target_i = target[mask]
            if source_i.size(0) > 0 and target_i.size(0) > 0:
                loss = loss + mmd_rbf(source_i, target_i)
                count += 1
    
    if count > 0:
        return loss / count
    return torch.tensor(0.0, device=source.device)


class L2D(Algorithm):
    """
    L2D: Learning to Diversify for Single Domain Generalization.
    
    Uses a learnable augmentation network to generate diverse samples while
    maintaining semantic consistency through contrastive learning.
    
    Hyperparameters:
        alpha1: Weight for contrastive loss (default: 1.0)
        alpha2: Weight for likelihood loss (default: 1.0)
        beta: Weight for CLUB divergence (default: 0.1)
        lr_aug: Learning rate for augmentation network (default: 10.0)
        aug_weight: Weight for augmented images in mixing (default: 0.6)
        variational_dim: Dimension of variational embedding (default: 512)
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
        self.alpha1 = hparams.get('alpha1', 1.0)
        self.alpha2 = hparams.get('alpha2', 1.0)
        self.beta = hparams.get('beta', 0.1)
        self.lr_aug = hparams.get('lr_aug', 10.0)
        self.aug_weight = hparams.get('aug_weight', 0.6)
        self.variational_dim = hparams.get('variational_dim', 512)
        
        # Get feature dimension from featurizer
        # Attempt to determine feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Variational head for reparameterization
        self.variational_head = VariationalHead(
            in_features=self.feature_dim,
            out_features=self.variational_dim
        )
        
        # Update classifier to use variational embedding dimension
        # We'll create a new classifier head
        self.var_classifier = nn.Linear(self.variational_dim, num_classes)
        
        # Augmentation network
        self.augnet = AugNet(image_size=224)
        
        # Contrastive loss
        self.contrastive_loss = SupConLoss(temperature=0.07)

        # ImageNet normalization (applied after augmentation)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Separate optimizer for augnet (set up in training script)
        self.augnet_optimizer = None
        
        # Step counter for alternating updates
        self.step_count = 0
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from featurizer."""
        # Try common attributes
        if hasattr(self.featurizer, 'n_outputs'):
            return self.featurizer.n_outputs
        if hasattr(self.featurizer, 'out_features'):
            return self.featurizer.out_features
        if hasattr(self.featurizer, 'num_features'):
            return self.featurizer.num_features
        # Default for ResNet18
        return 512
    
    def setup_augnet_optimizer(self):
        """Setup optimizer for augmentation network."""
        if self.augnet_optimizer is None:
            self.augnet_optimizer = torch.optim.SGD(
                self.augnet.parameters(),
                lr=self.lr_aug
            )
    
    def _normalize_images(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        # x should be in [0, 1] range after sigmoid
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std
    
    def generate_augmented(self, x: torch.Tensor, estimation: bool = False) -> torch.Tensor:
        """
        Generate augmented images using AugNet.
        
        Args:
            x: Original images
            estimation: Whether to use learned parameters
            
        Returns:
            Mixed augmented images
        """
        # Generate augmentation
        aug_output = self.augnet(x, estimation=estimation)
        
        # Apply sigmoid and normalize
        aug_normalized = self._normalize_images(torch.sigmoid(aug_output))
        
        # Mix with original
        mixed = self.aug_weight * aug_normalized + (1 - self.aug_weight) * x
        
        return mixed
    
    def forward_with_variational(self, x: torch.Tensor, train: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with variational encoding.
        """
        # Extract features
        features = self.featurizer(x)
        
        # Variational encoding
        var_dict = self.variational_head(features, train=train)
        
        # Classification using variational embedding
        logits = self.var_classifier(var_dict['embedding'])
        
        return {
            'logits': logits,
            'features': features,
            **var_dict
        }
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with L2D.
        
        Two-stage training:
        1. Train classifier with contrastive + likelihood loss
        2. Train augnet to maximize diversity
        """
        self.setup_augnet_optimizer()
        
        batch_size = x.size(0)
        
        # ============ Stage 1: Train Classifier ============
        # Generate augmented images
        x_aug = self.generate_augmented(x, estimation=False)
        
        # Combine original and augmented
        x_combined = torch.cat([x_aug, x], dim=0)
        y_combined = torch.cat([y, y], dim=0)
        
        # Forward pass
        out = self.forward_with_variational(x_combined, train=True)
        logits = out['logits']
        
        # Classification loss
        class_loss = self.compute_loss(logits, y_combined)
        
        # Contrastive loss
        # Embeddings: first half is augmented, second half is original
        emb_aug = F.normalize(out['embedding'][:batch_size], dim=1).unsqueeze(1)
        emb_src = F.normalize(out['embedding'][batch_size:], dim=1).unsqueeze(1)
        contrastive_features = torch.cat([emb_aug, emb_src], dim=1)
        con_loss = self.contrastive_loss(contrastive_features, y)
        
        # Likelihood loss (maximize likelihood of original given augmented)
        mu_aug = out['mu'][:batch_size]
        logvar_aug = out['logvar'][:batch_size]
        emb_orig = out['embedding'][batch_size:]
        likeli_loss = -loglikeli(mu_aug, logvar_aug, emb_orig)
        
        # Total classifier loss
        total_loss = class_loss + self.alpha1 * con_loss + self.alpha2 * likeli_loss
        
        # ============ Stage 2: Train AugNet ============
        # Generate augmented with estimation=True
        x_aug_est = self.generate_augmented(x, estimation=True)
        x_combined_est = torch.cat([x_aug_est, x], dim=0)
        
        # Forward pass (no grad for classifier)
        with torch.no_grad():
            out_est = self.forward_with_variational(x_combined_est, train=True)
        
        # Re-enable grad for augnet
        out_est_aug = self.forward_with_variational(x_aug_est, train=True)
        
        # CLUB divergence (minimize MI upper bound for diversity)
        mu_est = out_est_aug['mu']
        logvar_est = out_est_aug['logvar']
        emb_orig_detached = out['embedding'][batch_size:].detach()
        div_loss = club(mu_est, logvar_est, emb_orig_detached)
        
        # Conditional MMD for semantic consistency
        emb_aug_est = out_est_aug['embedding']
        mmd_loss = conditional_mmd_rbf(emb_aug_est, emb_orig_detached, y, self.num_classes)
        
        # AugNet loss: minimize MMD (consistency) + beta * maximize CLUB (diversity)
        augnet_loss = mmd_loss + self.beta * div_loss
        
        # Update augnet
        self.augnet_optimizer.zero_grad()
        augnet_loss.backward()
        self.augnet_optimizer.step()
        
        self.step_count += 1
        
        return {
            'loss': total_loss,
            'class_loss': class_loss,
            'contrastive_loss': con_loss,
            'likelihood_loss': likeli_loss,
            'augnet_loss': augnet_loss.detach(),
            'mmd_loss': mmd_loss.detach(),
            'div_loss': div_loss.detach(),
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (inference mode).
        """
        out = self.forward_with_variational(x, train=False)
        return out['logits']
    
    def get_trainable_params(self):
        """
        Get trainable parameters (excluding augnet which has its own optimizer).
        """
        return list(self.featurizer.parameters()) + \
               list(self.variational_head.parameters()) + \
               list(self.var_classifier.parameters())
