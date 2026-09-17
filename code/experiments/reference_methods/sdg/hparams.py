"""
Hyperparameter management for SDG (Single Domain Generalization) algorithms.

Each algorithm has specific hyperparameters with sensible defaults.
These can be overridden via command line using JSON format.
"""

import json
from typing import Dict, Any, Optional

# Default hyperparameters for each algorithm
HPARAMS_REGISTRY: Dict[str, Dict[str, Any]] = {
    # JiGen: Domain Generalization by Solving Jigsaw Puzzles
    'JiGen': {
        'jigsaw_n_classes': 30,       # Number of jigsaw permutation classes
        'jig_weight': 0.7,            # Weight for jigsaw puzzle loss
        'bias_whole_image': 0.9,      # Probability of showing unshuffled image
        'tile_random_grayscale': 0.1, # Probability of grayscaling a tile
    },
    
    # ADA: Adversarial Data Augmentation
    'ADA': {
        'gamma': 1.0,           # Weight for feature similarity regularization
        'T_adv': 15,            # Number of gradient ascent steps
        'lr_max': 1.0,          # Learning rate for gradient ascent on images
        'ada_frequency': 1,     # Apply ADA every N batches
        'epsilon': None,        # Max perturbation magnitude (None = no clipping)
    },
    
    # ME-ADA: Maximum-Entropy Adversarial Data Augmentation
    'MEADA': {
        'gamma': 1.0,           # Weight for feature similarity regularization
        'eta': 1.0,             # Weight for entropy maximization
        'T_adv': 15,            # Number of gradient ascent steps
        'lr_max': 20.0,         # Learning rate for gradient ascent on images
        'meada_frequency': 1,   # Apply ME-ADA every N batches
        'epsilon': None,        # Max perturbation magnitude (None = no clipping)
    },
    
    # L2D: Learning to Diversify
    'L2D': {
        'alpha1': 1.0,          # Weight for contrastive loss
        'alpha2': 1.0,          # Weight for likelihood loss
        'beta': 0.1,            # Weight for CLUB divergence
        'lr_aug': 10.0,         # Learning rate for augmentation network
        'aug_weight': 0.6,      # Weight for augmented images in mixing
        'variational_dim': 512, # Dimension of variational embedding
    },
    
    # ADVST: Adversarial Style Transfer
    'ADVST': {
        'gamma': 10.0,          # Weight for semantic distance regularization
        'eta': 10.0,            # Weight for entropy maximization in adversarial step
        'eta_min': 0.01,        # Weight for entropy regularization in minimization
        'beta': 1.0,            # Weight for contrastive loss
        'lr_max': 5.0,          # Learning rate for augmentation optimization
        'loops_adv': 50,        # Number of adversarial optimization steps
        'max_ops': 3,           # Maximum number of augmentation operations
        'use_contrastive': True, # Whether to use contrastive loss
        'pool_size': 5,         # Size of augmented data pool
    },
    
    # ACVC: Attention Consistency on Visual Corruptions
    'ACVC': {
        'lambd': 0.06,          # Weight for attention consistency loss
        'temperature': 1.0,     # Temperature for softmax in CAM
        'k_neg': 3,             # Number of top negative classes to minimize
        'corruption_prob': 0.5, # Probability of applying corruption
        'num_corruptions': 1,   # Number of corruptions per image
    },
    
    # CCSA: Contrastive Cross-domain Semantic Alignment
    'CCSA': {
        'margin': 1.0,          # Margin for contrastive loss
        'alpha': 0.25,          # Weight for CSA loss (1-alpha for classification)
        'aug_strength': 0.5,    # Strength of domain-shift augmentations
        'use_color_jitter': True,  # Use color jitter augmentation
        'use_grayscale': True,     # Use random grayscale conversion
        'use_noise': True,         # Use Gaussian noise augmentation
        'use_erasing': True,       # Use random erasing augmentation
    },
    
    # RSDA: Random Search Data Augmentation
    'RSDA': {
        'string_length': 3,     # Number of transformations to concatenate
        'pool_size': 50,        # Size of transformation pool
        'pool_prob': 0.5,       # Probability of sampling from pool vs random
        'search_iters': 5,      # Number of random search iterations per batch
        'use_pool': True,       # Whether to use transformation pool
    },
    
    # PDEN: Progressive Domain Expansion Network
    'PDEN': {
        'zdim': 10,             # Dimension of style vector
        'gen_hidden': 32,       # Hidden dimension of generator
        'w_cls': 1.0,           # Weight for classification loss on generated
        'w_con': 1.0,           # Weight for contrastive loss
        'w_div': 1.0,           # Weight for diversity loss
        'w_cyc': 10.0,          # Weight for cycle consistency loss
        'div_thresh': 0.1,      # Threshold for diversity loss
        'temperature': 0.07,    # Temperature for contrastive loss
        'lr_gen': 1e-3,         # Learning rate for generator
        'use_cycle': True,      # Whether to use cycle consistency
    },
    
    # MMLD: Mixture of Multiple Latent Domains
    'MMLD': {
        'num_domains': 3,       # Number of pseudo-domains
        'disc_hidden': 1024,    # Hidden dimension of discriminator
        'entropy_weight': 1.0,  # Weight for entropy regularization
        'grl_weight': 1.0,      # Maximum GRL weight
        'use_grl': True,        # Whether to use gradient reversal
        'progressive': True,    # Use progressive GRL/entropy scheduling
    },
    
    # MixStyle: Domain Generalization with MixStyle
    'MixStyle': {
        'p': 0.5,               # Probability of applying MixStyle
        'alpha': 0.1,           # Beta distribution parameter
        'mix': 'random',        # Mixing strategy: 'random' or 'crossdomain'
        'eps': 1e-6,            # Numerical stability constant
        'apply_to_features': True,  # Apply MixStyle to final features
    },
    
    # StyDeSty: Min-Max Stylization and Destylization
    'StyDeSty': {
        'alpha_feat_idt': 1.0,   # Weight for feature identity loss (backbone)
        'alpha_likelihood': 1.0, # Weight for likelihood loss (backbone)
        'beta_semantic': 1.0,    # Weight for semantic (MMD) loss (AugNet)
        'beta_likelihood': 0.1,  # Weight for CLUB likelihood loss (AugNet)
        'beta_feat_idt': 1.0,    # Weight for feature identity loss (AugNet)
        'lr_aug': 0.005,         # Learning rate for AugNet
        'aug_weight': 0.6,       # Weight for mixing augmented images
        'mid_feat': 256,         # Hidden dimension of ProbMLP
        'small_augnet': False,   # Use smaller AugNet for small images
    },
}


def get_hparams(
    algorithm_name: str,
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get hyperparameters for an algorithm with optional overrides.
    
    Args:
        algorithm_name: Name of the algorithm
        overrides: Dictionary of hyperparameter overrides
        
    Returns:
        Dictionary of hyperparameters
        
    Raises:
        ValueError: If algorithm_name is not recognized
    """
    if algorithm_name not in HPARAMS_REGISTRY:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: {list(HPARAMS_REGISTRY.keys())}"
        )
    
    # Start with defaults
    hparams = HPARAMS_REGISTRY[algorithm_name].copy()
    
    # Apply overrides if provided
    if overrides is not None:
        for key, value in overrides.items():
            if key in hparams:
                hparams[key] = value
            else:
                # Allow adding new hyperparameters (for flexibility)
                hparams[key] = value
    
    return hparams


def parse_hparams_string(hparams_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON string of hyperparameter overrides.
    
    Args:
        hparams_str: JSON string like '{"jig_weight": 0.5}'
        
    Returns:
        Dictionary of hyperparameters or None if input is None/empty
        
    Raises:
        ValueError: If the string is not valid JSON
    """
    if hparams_str is None or hparams_str.strip() == '':
        return None
    
    try:
        return json.loads(hparams_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid hyperparameter JSON: {hparams_str}. Error: {e}")
