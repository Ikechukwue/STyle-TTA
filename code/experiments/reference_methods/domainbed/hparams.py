"""
Hyperparameter management for DomainBed algorithms.

Each algorithm has specific hyperparameters with sensible defaults.
These can be overridden via command line using JSON format.
"""

import json
from typing import Dict, Any, Optional

# Default hyperparameters for each algorithm
HPARAMS_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ERM has no special hyperparameters
    'ERM': {},
    
    # Mixup: interpolation between samples
    'Mixup': {
        'mixup_alpha': 0.2,  # Beta distribution parameter
    },
    
    # RSC: Representation Self-Challenging
    'RSC': {
        'rsc_f_drop_factor': 0.33,  # Fraction of features to drop
        'rsc_b_drop_factor': 0.33,  # Fraction of batch samples affected
    },
    
    # SD: Spectral Decoupling
    'SD': {
        'sd_reg': 0.1,  # Regularization strength for logit penalty
    },
    
    # SelfReg: Self-supervised Contrastive Regularization
    # Note: Official implementation uses hardcoded values (no hparams)
    # lam ~ Beta(0.5, 0.5), feature weight = 0.3
    'SelfReg': {},
    
    # IB_ERM: Information Bottleneck
    'IB_ERM': {
        'ib_lambda': 1e-3,  # Weight for variance penalty
        'ib_penalty_anneal_iters': 500
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
        hparams_str: JSON string like '{"mixup_alpha": 0.4}'
        
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
