"""
DomainBed Domain Generalization Methods

This module provides domain-agnostic domain generalization algorithms
that can be used for single-domain training without requiring domain labels.

Supported algorithms:
- ERM: Empirical Risk Minimization (baseline)
- Mixup: Input mixup augmentation
- RSC: Representation Self-Challenging
- SD: Spectral Decoupling
- SelfReg: Self-supervised Contrastive Regularization
- IB_ERM: Information Bottleneck ERM
"""

from code.experiments.reference_methods.domainbed.algorithms import ALGORITHMS, get_algorithm
from code.experiments.reference_methods.domainbed.networks import create_featurizer_classifier, TimmFeaturizer, Classifier
from code.experiments.reference_methods.domainbed.hparams import get_hparams, HPARAMS_REGISTRY

__all__ = [
    'ALGORITHMS',
    'get_algorithm',
    'create_featurizer_classifier',
    'TimmFeaturizer',
    'Classifier',
    'get_hparams',
    'HPARAMS_REGISTRY',
]
