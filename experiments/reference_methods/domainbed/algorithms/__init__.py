"""
DomainBed Algorithm Registry

This module provides a registry of domain-agnostic domain generalization algorithms.
"""

from typing import Dict, Type, Any
import torch.nn as nn

from experiments.reference_methods.domainbed.algorithms.base import Algorithm
from experiments.reference_methods.domainbed.algorithms.erm import ERM
from experiments.reference_methods.domainbed.algorithms.mixup import Mixup
from experiments.reference_methods.domainbed.algorithms.rsc import RSC
from experiments.reference_methods.domainbed.algorithms.sd import SD
from experiments.reference_methods.domainbed.algorithms.selfreg import SelfReg
from experiments.reference_methods.domainbed.algorithms.ib_erm import IB_ERM

# Algorithm registry mapping names to classes
ALGORITHMS: Dict[str, Type[Algorithm]] = {
    'ERM': ERM,
    'Mixup': Mixup,
    'RSC': RSC,
    'SD': SD,
    'SelfReg': SelfReg,
    'IB_ERM': IB_ERM,
}


def get_algorithm(
    name: str,
    featurizer: nn.Module,
    classifier: nn.Module,
    num_classes: int,
    task_type: str,
    hparams: Dict[str, Any]
) -> Algorithm:
    """
    Factory function to create an algorithm instance.
    
    Args:
        name: Algorithm name (must be in ALGORITHMS registry)
        featurizer: Feature extractor network (backbone)
        classifier: Classification head
        num_classes: Number of output classes
        task_type: Task type ('multi-class' or 'multi-label')
        hparams: Algorithm-specific hyperparameters
        
    Returns:
        Algorithm instance
        
    Raises:
        ValueError: If algorithm name is not recognized
    """
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {name}. "
            f"Available algorithms: {list(ALGORITHMS.keys())}"
        )
    
    algorithm_class = ALGORITHMS[name]
    return algorithm_class(
        featurizer=featurizer,
        classifier=classifier,
        num_classes=num_classes,
        task_type=task_type,
        hparams=hparams
    )


__all__ = [
    'Algorithm',
    'ALGORITHMS',
    'get_algorithm',
    'ERM',
    'Mixup',
    'RSC',
    'SD',
    'SelfReg',
    'IB_ERM',
]
