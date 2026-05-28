"""FLUX.1-Kontext-dev text-guided style transfer (no velocity intervention).

This is a *baseline* variant that uses FLUX.1-Kontext-dev's image-editing
capability purely via text prompting — without the velocity-field intervention
of VeloEdit.  The content image is passed as the Kontext conditioning reference
and BLIP-2 is used to generate a style-applying edit prompt.

Comparison value: separates the contribution of the Kontext model's editing
capability from the velocity intervention introduced in ``veloedit``.
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional


class Method:
    """FLUX.1-Kontext-dev text-guided style transfer baseline."""

    def __init__(
        self,
        pretrained_weights: Optional[str] = None,
        device: str = "cpu",
        model_id: str = "black-forest-labs/FLUX.1-Kontext-dev",
        num_inference_steps: int = 28,
        guidance_scale: float = 2.5,
    ):
        self.device = device
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self._loaded = False
        self.pipe = None
        self._blip_processor = None
        self._blip_model = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load_pipeline(self, device: str) -> None:
        if self._loaded:
            return
        from diffusers import FluxKontextPipeline
        self.pipe = FluxKontextPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16
        ).to(device)
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
        """Apply Kontext text-guided style transfer.

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
            style_desc = self._describe_image(s_pil, device)
            prompt = f"Repaint this image in the following artistic style: {style_desc}"

            out_pil = self.pipe(
                image=c_pil,
                prompt=prompt,
                height=size,
                width=size,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(device=device).manual_seed(42),
                output_type="pil",
            ).images[0]
            results.append(self._pil_to_tensor(out_pil).squeeze(0).float())

        return torch.stack(results).clamp(0, 1).to(content.device)
