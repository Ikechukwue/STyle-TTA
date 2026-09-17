"""
RetriStyle Diffusion Sub-package
=================================

Customised StyleID diffusion component with **overloaded attention inputs**.

The modified U-Net Self-Attention mechanism can:
1. Accept full reference *images* (encoding them to extract K/V features).
2. Directly accept pre-computed K and V tensors (bypassing the encoder for
   cached styles).

The test (content) image always provides the Query (Q).
"""

from .style_injection import StyleInjectionDiffusion

__all__ = ["StyleInjectionDiffusion"]
