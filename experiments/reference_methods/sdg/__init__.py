"""
Single Domain Generalization (SDG) Methods

This module provides implementations of state-of-the-art SDG methods
for training classifiers with improved domain generalization.

Supported algorithms:
- JiGen: Domain Generalization by Solving Jigsaw Puzzles (CVPR 2019)
"""

from experiments.reference_methods.sdg.algorithms import ALGORITHMS, get_algorithm
from experiments.reference_methods.sdg.hparams import get_hparams, parse_hparams_string

__all__ = [
    'ALGORITHMS',
    'get_algorithm', 
    'get_hparams',
    'parse_hparams_string',
]
