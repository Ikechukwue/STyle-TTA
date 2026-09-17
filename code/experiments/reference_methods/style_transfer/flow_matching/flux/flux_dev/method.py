"""FLUX.1-dev style transfer.

FLUX.1-dev is Black Forest Labs' main flow-matching model.  Style is injected
via the XLabs-AI IP-Adapter when available, with BLIP-2 text prompting as
fallback.  Guidance scale follows the official FLUX.1-dev recommendation (3.5).
"""

from code.experiments.reference_methods.style_transfer.flow_matching.flux.base import (
    FluxIPAdapterStyleBase,
)


class Method(FluxIPAdapterStyleBase):
    """FLUX.1-dev style transfer (28 steps, guidance_scale=3.5)."""

    MODEL_ID = "black-forest-labs/FLUX.1-dev"
    NUM_STEPS_DEFAULT = 28
    GUIDANCE_SCALE_DEFAULT = 3.5
