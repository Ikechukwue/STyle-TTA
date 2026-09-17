"""
SDG Algorithm Registry

This module provides a registry of single domain generalization algorithms.
"""

from typing import Dict, Type, Any
import torch.nn as nn

from code.experiments.reference_methods.sdg.algorithms.base import Algorithm
from code.experiments.reference_methods.sdg.algorithms.jigen import JiGen
from code.experiments.reference_methods.sdg.algorithms.ada import ADA
from code.experiments.reference_methods.sdg.algorithms.meada import MEADA
from code.experiments.reference_methods.sdg.algorithms.l2d import L2D
from code.experiments.reference_methods.sdg.algorithms.advst import ADVST
from code.experiments.reference_methods.sdg.algorithms.acvc import ACVC
from code.experiments.reference_methods.sdg.algorithms.ccsa import CCSA
from code.experiments.reference_methods.sdg.algorithms.rsda import RSDA
from code.experiments.reference_methods.sdg.algorithms.pden import PDEN
from code.experiments.reference_methods.sdg.algorithms.mmld import MMLD
from code.experiments.reference_methods.sdg.algorithms.mixstyle import MixStyle
from code.experiments.reference_methods.sdg.algorithms.stydesty import StyDeSty

# Algorithm registry mapping names to classes
ALGORITHMS: Dict[str, Type[Algorithm]] = {
    'JiGen': JiGen,
    'ADA': ADA,
    'MEADA': MEADA,
    'L2D': L2D,
    'ADVST': ADVST,
    'ACVC': ACVC,
    'CCSA': CCSA,
    'RSDA': RSDA,
    'PDEN': PDEN,
    'MMLD': MMLD,
    'MixStyle': MixStyle,
    'StyDeSty': StyDeSty,
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
