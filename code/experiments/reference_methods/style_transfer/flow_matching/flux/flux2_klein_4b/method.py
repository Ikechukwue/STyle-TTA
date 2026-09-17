"""
xAILab Bamberg
University of Bamberg

@description:
FLUX.2-Klein-4B style transfer reference method.

FLUX.2-Klein-4B is the smaller of the two FLUX.2 Klein distilled models
(HF: black-forest-labs/FLUX.2-klein-4B).  Faster inference than FLUX.2-dev
at some quality cost.  Style injection delegates to FluxIPAdapterStyleBase.

@author: Sebastian Doerrich
"""

from code.experiments.reference_methods.style_transfer.flow_matching.flux.base import (
    FluxIPAdapterStyleBase,
)


class Method(FluxIPAdapterStyleBase):
    """FLUX.2-Klein-4B style transfer (28 steps, guidance_scale=3.5)."""

    # TODO: verify pipeline class for FLUX.2-Klein models (may need FluxPipeline).
    MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
    NUM_STEPS_DEFAULT = 28
    GUIDANCE_SCALE_DEFAULT = 3.5
