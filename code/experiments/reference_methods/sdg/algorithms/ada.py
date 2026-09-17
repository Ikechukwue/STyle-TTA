"""
ADA: Adversarial Data Augmentation

Reference:
    Volpi et al., "Generalizing to Unseen Domains via Adversarial Data Augmentation", NeurIPS 2018
    https://github.com/ricvolpi/generalize-unseen-domains

ADA creates adversarial examples by performing gradient ascent on input images to
maximize classification loss while keeping the learned features similar to the original.
This creates "hard" augmented examples that improve domain generalization.

Loss formulation:
    - Minimizer (network weights): minimize cross-entropy loss
    - Maximizer (image pixels): maximize (cross-entropy - gamma * feature_similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from code.experiments.reference_methods.sdg.algorithms.base import Algorithm


class ADA(Algorithm):
    """
    ADA: Adversarial Data Augmentation.
    
    Creates adversarial augmentations by gradient ascent on input images.
    The adversarial images maximize classification loss while maintaining
    feature similarity to the original images.
    
    Hyperparameters:
        gamma: Weight for feature similarity regularization (default: 1.0)
        T_adv: Number of gradient ascent steps on images (default: 15)
        lr_max: Learning rate for gradient ascent on images (default: 1.0)
        ada_frequency: Apply ADA every N batches (default: 1, i.e., every batch)
        epsilon: Maximum perturbation magnitude (default: None, no clipping)
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
        self.gamma = hparams.get('gamma', 1.0)
        self.T_adv = hparams.get('T_adv', 15)
        self.lr_max = hparams.get('lr_max', 1.0)
        self.ada_frequency = hparams.get('ada_frequency', 1)
        self.epsilon = hparams.get('epsilon', None)  # Optional perturbation clipping
        
        # Counter for frequency-based application
        self.register_buffer('step_count', torch.tensor([0]))
    
    def _generate_adversarial(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples via gradient ascent on input images.
        
        The objective is to maximize: L_class - gamma * L_feature_sim
        where L_class is classification loss and L_feature_sim is MSE between
        original and perturbed features.
        
        Args:
            x: Original input images
            y: Ground truth labels
            
        Returns:
            Adversarial images (detached from computation graph)
        """
        # Get original features (detached, as target for similarity)
        with torch.no_grad():
            original_features = self.featurizer(x).detach()
        
        # Initialize adversarial images as a copy of original
        x_adv = x.clone().detach().requires_grad_(True)
        
        for _ in range(self.T_adv):
            # Forward pass with adversarial images
            features_adv = self.featurizer(x_adv)
            logits_adv = self.classifier(features_adv)
            
            # Compute classification loss (to maximize)
            class_loss = self.compute_loss(logits_adv, y)
            
            # Compute feature similarity loss (to minimize, so subtract)
            # MSE between original and adversarial features
            feature_sim_loss = F.mse_loss(features_adv, original_features)
            
            # Adversarial objective: maximize class_loss - gamma * feature_sim
            # We negate because we'll use gradients for ascent
            adv_loss = class_loss - self.gamma * feature_sim_loss
            
            # Compute gradients w.r.t. adversarial images
            grad = torch.autograd.grad(adv_loss, x_adv, create_graph=False)[0]
            
            # Gradient ascent step
            with torch.no_grad():
                x_adv = x_adv + self.lr_max * grad
                
                # Optional: clip perturbation magnitude
                if self.epsilon is not None:
                    perturbation = x_adv - x
                    perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                    x_adv = x + perturbation
                
                # Clip to valid image range [0, 1] or maintain input range
                # Note: inputs are normalized, so we don't clip here
                x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv.detach()
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with adversarial data augmentation.
        
        Args:
            x: Input images
            y: Ground truth class labels
            
        Returns:
            Dictionary with 'loss', 'class_loss', and optionally 'adv_loss'
        """
        self.step_count += 1
        
        # Decide whether to apply ADA this step
        apply_ada = (self.step_count % self.ada_frequency == 0)
        
        if apply_ada and self.training:
            # Generate adversarial examples
            x_adv = self._generate_adversarial(x, y)
            
            # Combine original and adversarial examples
            x_combined = torch.cat([x, x_adv], dim=0)
            y_combined = torch.cat([y, y], dim=0)
            
            # Forward pass on combined batch
            features = self.featurizer(x_combined)
            logits = self.classifier(features)
            
            # Compute loss on combined batch
            loss = self.compute_loss(logits, y_combined)
            
            # Also compute loss on original only for logging
            with torch.no_grad():
                features_orig = self.featurizer(x)
                logits_orig = self.classifier(features_orig)
                class_loss = self.compute_loss(logits_orig, y)
            
            return {
                'loss': loss,
                'class_loss': class_loss,
                'ada_applied': torch.tensor(1.0)
            }
        else:
            # Standard forward pass without ADA
            features = self.featurizer(x)
            logits = self.classifier(features)
            loss = self.compute_loss(logits, y)
            
            return {
                'loss': loss,
                'class_loss': loss,
                'ada_applied': torch.tensor(0.0)
            }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for input images (no adversarial augmentation at test time).
        
        Args:
            x: Input images
            
        Returns:
            Class logits
        """
        return self.network(x)
