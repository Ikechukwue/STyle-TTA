"""
Network architectures for SDG algorithms.

This module re-exports the DomainBed network utilities for consistency.
SDG algorithms use the same featurizer-classifier decomposition pattern.
"""

# Re-export from DomainBed for consistency
from code.experiments.reference_methods.domainbed.networks import (
    TimmFeaturizer,
    Classifier,
    FeaturizerClassifierNetwork,
    create_featurizer_classifier,
    get_network_state_dict_for_timm,
)

__all__ = [
    'TimmFeaturizer',
    'Classifier', 
    'FeaturizerClassifierNetwork',
    'create_featurizer_classifier',
    'get_network_state_dict_for_timm',
]
