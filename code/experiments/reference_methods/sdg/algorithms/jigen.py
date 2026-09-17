"""
JiGen: Domain Generalization by Solving Jigsaw Puzzles

Reference:
    Carlucci et al., "Domain Generalization by Solving Jigsaw Puzzles", CVPR 2019
    https://github.com/fmcarlucci/JigenDG

JiGen uses a jigsaw puzzle solving task as a self-supervised auxiliary task
to learn domain-invariant representations. The image is split into a 3x3 grid,
tiles are shuffled according to predefined permutations, and the network must
predict which permutation was used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from itertools import permutations

from code.experiments.reference_methods.sdg.algorithms.base import Algorithm


def generate_permutations(n_classes: int, seed: int = 42) -> np.ndarray:
    """
    Generate a set of jigsaw puzzle permutations.
    
    Uses a greedy algorithm to select permutations with maximum hamming distance
    to ensure the puzzle classes are sufficiently different.
    
    Args:
        n_classes: Number of permutation classes to generate
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_classes, 9) containing permutations
    """
    np.random.seed(seed)
    
    # All possible permutations of 9 tiles
    all_perms = list(permutations(range(9)))
    all_perms = np.array(all_perms)
    
    # Shuffle to randomize selection
    indices = np.random.permutation(len(all_perms))
    all_perms = all_perms[indices]
    
    # Greedy selection based on hamming distance
    selected = [all_perms[0]]
    all_perms = all_perms[1:]
    
    while len(selected) < n_classes and len(all_perms) > 0:
        # Compute minimum hamming distance to already selected permutations
        min_distances = []
        for perm in all_perms:
            distances = [np.sum(perm != s) for s in selected]
            min_distances.append(min(distances))
        
        # Select permutation with maximum minimum distance
        best_idx = np.argmax(min_distances)
        selected.append(all_perms[best_idx])
        all_perms = np.delete(all_perms, best_idx, axis=0)
    
    return np.array(selected)


class JiGen(Algorithm):
    """
    JiGen: Domain Generalization by Solving Jigsaw Puzzles.
    
    Trains a dual-head network where one head predicts class labels and
    another head predicts which jigsaw permutation was applied to the input.
    
    Hyperparameters:
        jigsaw_n_classes: Number of jigsaw permutation classes (default: 30)
        jig_weight: Weight for jigsaw loss (default: 0.7)
        bias_whole_image: Probability of showing unshuffled image (default: 0.9)
        tile_random_grayscale: Probability of converting a tile to grayscale (default: 0.1)
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
        self.jigsaw_n_classes = hparams.get('jigsaw_n_classes', 30)
        self.jig_weight = hparams.get('jig_weight', 0.7)
        self.bias_whole_image = hparams.get('bias_whole_image', 0.9)
        self.tile_random_grayscale = hparams.get('tile_random_grayscale', 0.1)
        
        # Grid configuration
        self.grid_size = 3
        self.n_tiles = self.grid_size ** 2
        
        # Generate permutations on-the-fly
        # +1 for class 0 which represents the unshuffled (original) image
        self.register_buffer(
            'permutations',
            torch.from_numpy(generate_permutations(self.jigsaw_n_classes)).long()
        )
        
        # Jigsaw classifier head
        # +1 class for unshuffled image (class 0)
        n_features = featurizer.n_outputs
        self.jigsaw_classifier = nn.Linear(n_features, self.jigsaw_n_classes + 1)
    
    def _tile_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split image into a grid of tiles.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Tiles of shape (B, n_tiles, C, tile_h, tile_w)
        """
        B, C, H, W = x.shape
        
        # Compute tile size (may need padding for non-divisible sizes)
        tile_h = H // self.grid_size
        tile_w = W // self.grid_size
        
        # Handle case where image size is not perfectly divisible
        # Crop to nearest divisible size
        H_crop = tile_h * self.grid_size
        W_crop = tile_w * self.grid_size
        
        if H_crop != H or W_crop != W:
            # Center crop
            start_h = (H - H_crop) // 2
            start_w = (W - W_crop) // 2
            x = x[:, :, start_h:start_h + H_crop, start_w:start_w + W_crop]
        
        # Reshape to grid
        # (B, C, H, W) -> (B, C, grid_size, tile_h, grid_size, tile_w)
        x = x.view(B, C, self.grid_size, tile_h, self.grid_size, tile_w)
        
        # Permute to (B, grid_size, grid_size, C, tile_h, tile_w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        # Reshape to (B, n_tiles, C, tile_h, tile_w)
        x = x.reshape(B, self.n_tiles, C, tile_h, tile_w)
        
        return x
    
    def _reassemble_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Reassemble tiles back into images.
        
        Args:
            tiles: Tiles of shape (B, n_tiles, C, tile_h, tile_w)
            
        Returns:
            Images of shape (B, C, H, W)
        """
        B, n_tiles, C, tile_h, tile_w = tiles.shape
        
        # Reshape to (B, grid_size, grid_size, C, tile_h, tile_w)
        tiles = tiles.view(B, self.grid_size, self.grid_size, C, tile_h, tile_w)
        
        # Permute to (B, C, grid_size, tile_h, grid_size, tile_w)
        tiles = tiles.permute(0, 3, 1, 4, 2, 5)
        
        # Reshape to (B, C, H, W)
        H = self.grid_size * tile_h
        W = self.grid_size * tile_w
        tiles = tiles.reshape(B, C, H, W)
        
        return tiles
    
    def _apply_tile_augmentation(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Apply random grayscale augmentation to tiles.
        
        Args:
            tiles: Tiles of shape (B, n_tiles, C, tile_h, tile_w)
            
        Returns:
            Augmented tiles
        """
        if self.tile_random_grayscale <= 0 or not self.training:
            return tiles
        
        B, n_tiles, C, tile_h, tile_w = tiles.shape
        
        # Random mask for grayscale conversion
        mask = torch.rand(B, n_tiles, 1, 1, 1, device=tiles.device) < self.tile_random_grayscale
        
        # Convert to grayscale (average across channels, then expand)
        gray = tiles.mean(dim=2, keepdim=True).expand_as(tiles)
        
        # Apply mask
        tiles = torch.where(mask.expand_as(tiles), gray, tiles)
        
        return tiles
    
    def _create_jigsaw(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create jigsaw puzzles from batch of images.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Tuple of:
                - Shuffled/reassembled images of shape (B, C, H', W')
                - Jigsaw labels of shape (B,) indicating permutation class
        """
        B = x.shape[0]
        device = x.device
        
        # Split into tiles
        tiles = self._tile_image(x)  # (B, n_tiles, C, tile_h, tile_w)
        
        # Apply tile augmentation (random grayscale)
        tiles = self._apply_tile_augmentation(tiles)
        
        # Generate random permutation orders for each sample
        # Order 0 = unshuffled, Order 1 to jigsaw_n_classes = shuffled
        if self.training:
            orders = torch.randint(
                0, self.jigsaw_n_classes + 1, 
                (B,), 
                device=device
            )
            
            # Bias towards showing whole image (unshuffled)
            if self.bias_whole_image > 0:
                bias_mask = torch.rand(B, device=device) < self.bias_whole_image
                orders = torch.where(bias_mask, torch.zeros_like(orders), orders)
        else:
            # During evaluation, always use unshuffled
            orders = torch.zeros(B, dtype=torch.long, device=device)
        
        # Apply permutations
        shuffled_tiles = tiles.clone()
        
        for b in range(B):
            order = orders[b].item()
            if order > 0:
                # Get permutation (order-1 because order 0 is unshuffled)
                perm = self.permutations[order - 1]
                shuffled_tiles[b] = tiles[b, perm]
        
        # Reassemble into images
        jigsaw_images = self._reassemble_tiles(shuffled_tiles)
        
        return jigsaw_images, orders
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform one training step with jigsaw puzzle auxiliary task.
        
        Args:
            x: Input images
            y: Ground truth class labels
            
        Returns:
            Dictionary with 'loss', 'class_loss', and 'jigsaw_loss'
        """
        # Create jigsaw puzzles
        jigsaw_images, jigsaw_labels = self._create_jigsaw(x)
        
        # Forward pass through featurizer
        features = self.featurizer(jigsaw_images)
        
        # Classification predictions
        class_logits = self.classifier(features)
        
        # Jigsaw predictions
        jigsaw_logits = self.jigsaw_classifier(features)
        
        # Compute losses
        class_loss = self.compute_loss(class_logits, y)
        jigsaw_loss = F.cross_entropy(jigsaw_logits, jigsaw_labels)
        
        # Combined loss
        total_loss = class_loss + self.jig_weight * jigsaw_loss
        
        return {
            'loss': total_loss,
            'class_loss': class_loss,
            'jigsaw_loss': jigsaw_loss
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for input images (without jigsaw augmentation).
        
        At inference time, we use the original images without jigsaw shuffling.
        
        Args:
            x: Input images
            
        Returns:
            Class logits
        """
        # Use original images (no jigsaw) for prediction
        return self.network(x)
