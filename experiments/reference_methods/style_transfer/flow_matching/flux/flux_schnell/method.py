"""FLUX.1-schnell style transfer.

FLUX.1-schnell is the fastest FLUX variant (4 distilled steps, Apache-2.0).
Because it is a distilled model, guidance_scale must be 0.0.  No HF token is
required.  Style is injected via the XLabs-AI IP-Adapter when available,
otherwise via BLIP-2 text prompt alone.

Useful as a fast prototyping baseline before running heavier methods.
"""

from experiments.reference_methods.style_transfer.flow_matching.flux.base import (
    FluxIPAdapterStyleBase,
)


class Method(FluxIPAdapterStyleBase):
    """FLUX.1-schnell style transfer (4-step distilled, no HF token needed)."""

    MODEL_ID = "black-forest-labs/FLUX.1-schnell"
    NUM_STEPS_DEFAULT = 4
    GUIDANCE_SCALE_DEFAULT = 0.0
