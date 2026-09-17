"""
ME-ADA: Maximum-Entropy Adversarial Data Augmentation

Reference:
    Zhao et al., "Maximum-Entropy Adversarial Data Augmentation for Improved 
    Generalization and Robustness", NeurIPS 2020
    https://github.com/garyzhao/ME-ADA

ME-ADA extends ADA by adding an entropy maximization term to the adversarial objective.
This creates adversarial examples that not only fool the classifier but also 
maximize prediction uncertainty (entropy).

Loss formulation:
    - Minimizer (network weights): minimize cross-entropy loss
    - Maximizer (image pixels): maximize (class_loss + eta * entropy - gamma * feature_similarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from code.experiments.reference_methods.sdg.algorithms.base import Algorithm


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of predictions.
    
    Entropy H = -sum(p * log(p)) where p = softmax(logits)
    
    Args:
        logits: Raw logits from classifier
        
    Returns:
        Mean entropy across batch
    """
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -1.0 * (probs * log_probs).sum(dim=1)
    return entropy.mean()


class MEADA(Algorithm):
    """
    ME-ADA: Maximum-Entropy Adversarial Data Augmentation.
    
    Extends ADA by adding entropy maximization to create adversarial examples
    that are both misclassified and have high prediction uncertainty.
    
    Hyperparameters:
        gamma: Weight for feature similarity regularization (default: 1.0)
        eta: Weight for entropy maximization (default: 1.0)
        T_adv: Number of gradient ascent steps on images (default: 15)
        lr_max: Learning rate for gradient ascent on images (default: 20.0)
        meada_frequency: Apply ME-ADA every N batches (default: 1)
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
        self.eta = hparams.get('eta', 1.0)
        self.T_adv = hparams.get('T_adv', 15)
        self.lr_max = hparams.get('lr_max', 20.0)
        self.meada_frequency = hparams.get('meada_frequency', 1)
        self.epsilon = hparams.get('epsilon', None)
        
        # Counter for frequency-based application
        self.register_buffer('step_count', torch.tensor([0]))
    
    def _generate_adversarial(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial examples via gradient ascent with entropy maximization.
        
        The objective is to maximize: L_class + eta * entropy - gamma * feature_sim
        where:
            - L_class is classification loss
            - entropy is prediction entropy (higher = more uncertain)
            - feature_sim is MSE between original and perturbed features
        
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
            
            # Compute entropy (to maximize for more uncertain predictions)
            pred_entropy = entropy_loss(logits_adv)
            
            # Compute feature similarity loss (to minimize, so subtract)
            feature_sim_loss = F.mse_loss(features_adv, original_features)
            
            # ME-ADA objective: maximize (class_loss + eta * entropy - gamma * feature_sim)
            adv_loss = class_loss + self.eta * pred_entropy - self.gamma * feature_sim_loss
            
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
                
                x_adv = x_adv.detach().requires_grad_(True)
        
        return x_adv.detach()
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with maximum-entropy adversarial data augmentation.
        
        Args:
            x: Input images
            y: Ground truth class labels
            
        Returns:
            Dictionary with 'loss', 'class_loss', and 'meada_applied'
        """
        self.step_count += 1
        
        # Decide whether to apply ME-ADA this step
        apply_meada = (self.step_count % self.meada_frequency == 0)
        
        if apply_meada and self.training:
            # Generate adversarial examples with entropy maximization
            x_adv = self._generate_adversarial(x, y)
            
            # Combine original and adversarial examples
            x_combined = torch.cat([x, x_adv], dim=0)
            y_combined = torch.cat([y, y], dim=0)
            
            # Forward pass on combined batch
            features = self.featurizer(x_combined)
            logits = self.classifier(features)
            
            # Compute loss on combined batch
            loss = self.compute_loss(logits, y_combined)
            
            # Compute loss on original only for logging
            with torch.no_grad():
                features_orig = self.featurizer(x)
                logits_orig = self.classifier(features_orig)
                class_loss = self.compute_loss(logits_orig, y)
            
            return {
                'loss': loss,
                'class_loss': class_loss,
                'meada_applied': torch.tensor(1.0)
            }
        else:
            # Standard forward pass without ME-ADA
            features = self.featurizer(x)
            logits = self.classifier(features)
            loss = self.compute_loss(logits, y)
            
            return {
                'loss': loss,
                'class_loss': loss,
                'meada_applied': torch.tensor(0.0)
            }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for input images.
        
        Args:
            x: Input images
            
        Returns:
            Class logits
        """
        return self.network(x)
