"""Base class for FLUX-family style-transfer methods.

All variants (dev, Krea-dev, schnell) share the same FluxPipeline API and the
same IP-Adapter-based style injection approach. Subclasses only need to override
MODEL_ID (and optionally NUM_STEPS_DEFAULT / GUIDANCE_SCALE_DEFAULT).

Style injection strategy
------------------------
1. BLIP-2 captions the *content* image to build the generation prompt.
2. The *style* image is passed as ``ip_adapter_image`` if the XLabs-AI
   ``flux-ip-adapter`` can be loaded; otherwise the method falls back to
   generating from the text prompt alone (model still produces reasonable output
   because the FLUX base models are very strong text-to-image generators).
3. The pipeline generates from pure noise so the output may differ from the
   content in layout; for layout-preservation experiments prefer ``flux_kontext``
   or ``veloedit``.
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional


class FluxIPAdapterStyleBase:
    """FLUX FluxPipeline + optional IP-Adapter for style transfer."""

    # ---- Subclass overrides ------------------------------------------------
    MODEL_ID: str = "black-forest-labs/FLUX.1-dev"
    NUM_STEPS_DEFAULT: int = 28
    GUIDANCE_SCALE_DEFAULT: float = 3.5
    IP_ADAPTER_REPO: str = "XLabs-AI/flux-ip-adapter"
    IP_ADAPTER_WEIGHTS: str = "ip_adapter.safetensors"
    # -------------------------------------------------------------------------

    def __init__(
        self,
        pretrained_weights: Optional[str] = None,
        device: str = "cpu",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        ip_adapter_scale: float = 0.6,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps or self.NUM_STEPS_DEFAULT
        self.guidance_scale = guidance_scale if guidance_scale is not None else self.GUIDANCE_SCALE_DEFAULT
        self.ip_adapter_scale = ip_adapter_scale
        self._loaded = False
        self._has_ip_adapter = False
        self.pipe = None
        self._blip_processor = None
        self._blip_model = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_pipeline(self, device: str) -> None:
        if self._loaded:
            return
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16
        ).to(device)
        try:
            self.pipe.load_ip_adapter(
                self.IP_ADAPTER_REPO,
                weight_name=self.IP_ADAPTER_WEIGHTS,
            )
            self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            self._has_ip_adapter = True
        except Exception:
            self._has_ip_adapter = False
        self._loaded = True

    def _load_blip(self, device: str) -> None:
        if self._blip_model is not None:
            return
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        mid = "Salesforce/blip2-flan-t5-xl"
        self._blip_processor = AutoProcessor.from_pretrained(mid)
        self._blip_model = Blip2ForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.float16
        ).to(device)
        self._blip_model.eval()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def get_native_image_size() -> int:
        return 1024

    def _describe_image(self, pil_img: Image.Image, device: str) -> str:
        """Return a short BLIP-2 caption for *pil_img*."""
        self._load_blip(device)
        inputs = self._blip_processor(images=pil_img, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            ids = self._blip_model.generate(**inputs, max_new_tokens=50)
        return self._blip_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    @staticmethod
    def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
        arr = t.squeeze(0).permute(1, 2, 0).cpu().float().clamp(0, 1).numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply FLUX style transfer.

        Args:
            content: (B, 3, H, W) in [0, 1].
            style:   (B, 3, H, W) in [0, 1].

        Returns:
            Stylized tensor (B, 3, H, W) in [0, 1].
        """
        device = str(content.device)
        self._load_pipeline(device)

        size = self.get_native_image_size()
        if content.shape[-2:] != (size, size):
            content = torch.nn.functional.interpolate(content, (size, size), mode="bilinear", align_corners=False)
        if style.shape[-2:] != (size, size):
            style = torch.nn.functional.interpolate(style, (size, size), mode="bilinear", align_corners=False)

        results = []
        for i in range(content.shape[0]):
            c_pil = self._tensor_to_pil(content[i : i + 1])
            s_pil = self._tensor_to_pil(style[i : i + 1])
            content_desc = self._describe_image(c_pil, device)
            style_desc = self._describe_image(s_pil, device)
            prompt = f"{content_desc}, in the style of {style_desc}"

            gen_kwargs: dict = dict(
                prompt=prompt,
                height=size,
                width=size,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(device=device).manual_seed(42),
                output_type="pil",
            )
            if self._has_ip_adapter:
                gen_kwargs["ip_adapter_image"] = s_pil

            out_pil = self.pipe(**gen_kwargs).images[0]
            results.append(self._pil_to_tensor(out_pil).squeeze(0).float())

        return torch.stack(results).clamp(0, 1).to(content.device)
