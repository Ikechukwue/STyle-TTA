"""
xAILab Bamberg
University of Bamberg

@description:
FLUX.2-dev style transfer reference method.

FLUX.2-dev is Black Forest Labs' second-generation image-to-image flow-matching
model (HF: black-forest-labs/FLUX.2-dev). It supports Kontext-style image
conditioning via FluxKontextPipeline.  Style injection uses the XLabs-AI
IP-Adapter when available, with BLIP-2 text prompting as fallback.

@author: Sebastian Doerrich
"""

from experiments.reference_methods.style_transfer.flow_matching.flux.base import (
    FluxIPAdapterStyleBase,
)


class Method(FluxIPAdapterStyleBase):
    """FLUX.2-dev style transfer (28 steps, guidance_scale=3.5)."""

    # TODO: verify pipeline class — FluxKontextPipeline if image-conditioned,
    #       else FluxPipeline.  FLUX.2-dev ships as image-to-image on HF.
    MODEL_ID = "black-forest-labs/FLUX.2-dev"
    NUM_STEPS_DEFAULT = 28
    GUIDANCE_SCALE_DEFAULT = 3.5
