"""
TTA Baseline Methods
====================

Reference implementations for standard Test-Time Augmentation baselines:

- GeometricTTA: 16-view geometric augmentations (crops, flips, rotations)
- TENT: Test-time Entropy Minimization (updates BatchNorm affine parameters)
- AdaIN-TTA: Global statistical matching with randomly queried training references
"""

from .geometric_tta import GeometricTTA
from .tent_tta import TENT

__all__ = ["GeometricTTA", "TENT"]
