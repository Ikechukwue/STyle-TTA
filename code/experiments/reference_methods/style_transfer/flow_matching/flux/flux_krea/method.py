"""FLUX.1-Krea-dev style transfer.

FLUX.1-Krea-dev is a drop-in replacement for FLUX.1-dev fine-tuned for
aesthetic photography and visual quality.  Uses the same FluxPipeline API with
guidance_scale=4.5 (Krea's recommended value).  Style is injected via the
XLabs-AI IP-Adapter when available, otherwise via BLIP-2 text prompt.
"""

from code.experiments.reference_methods.style_transfer.flow_matching.flux.base import (
    FluxIPAdapterStyleBase,
)


class Method(FluxIPAdapterStyleBase):
    """FLUX.1-Krea-dev style transfer (aesthetic, higher guidance)."""

    MODEL_ID = "black-forest-labs/FLUX.1-Krea-dev"
    NUM_STEPS_DEFAULT = 28
    GUIDANCE_SCALE_DEFAULT = 4.5
