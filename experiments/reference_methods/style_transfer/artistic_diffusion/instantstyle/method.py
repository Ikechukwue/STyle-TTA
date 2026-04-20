"""
InstantStyle – Free Lunch towards Style-Preserving in Text-to-Image Generation.

Training-free style transfer using SDXL + IP-Adapter with style-block-only injection
and ControlNet (canny) for content preservation.

Paper : arXiv 2404.02733 (2024)
Repo  : https://github.com/instantX-research/InstantStyle
Docs  : https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter

Required HuggingFace models (auto-downloaded):
  - stabilityai/stable-diffusion-xl-base-1.0
  - h94/IP-Adapter  (sdxl_models/ip-adapter_sdxl.bin + image_encoder)
  - diffusers/controlnet-canny-sdxl-1.0

GPU : ~16-24 GB VRAM (fp16)
"""

import cv2
import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)


class Method:
    """InstantStyle: style transfer via IP-Adapter style-block injection + ControlNet."""

    def __init__(self, device="cuda"):
        """
        Parameters
        ----------
        device : str
        """
        self.device = device
        self.cfg = self.get_default_config()
        self._initialize_network()

    @staticmethod
    def get_default_config():
        return SimpleNamespace(
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            ip_adapter_repo="h94/IP-Adapter",
            ip_adapter_subfolder="sdxl_models",
            ip_adapter_weight="ip-adapter_sdxl.bin",
            controlnet_model="diffusers/controlnet-canny-sdxl-1.0",
            num_inference_steps=30,
            guidance_scale=5.0,
            controlnet_conditioning_scale=0.6,
            ip_adapter_scale=1.0,
            canny_low=100,
            canny_high=200,
            prompt="masterpiece, best quality",
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        )

    @staticmethod
    def get_native_image_size():
        return 1024  # SDXL native

    # ── network setup ──────────────────────────────────────────────────
    def _initialize_network(self):
        cfg = self.cfg

        controlnet = ControlNetModel.from_pretrained(
            cfg.controlnet_model, torch_dtype=torch.float16,
        )

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
        )

        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            cfg.base_model,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
        ).to(self.device)

        # Load IP-Adapter
        self.pipe.load_ip_adapter(
            cfg.ip_adapter_repo,
            subfolder=cfg.ip_adapter_subfolder,
            weight_name=cfg.ip_adapter_weight,
        )

        # Set IP-Adapter scale to inject ONLY into the style block
        # up_blocks.0.attentions.1 = style block (index 1 in up.block_0)
        scale = {
            "down": {"block_2": [0.0, 0.0]},
            "up": {"block_0": [0.0, cfg.ip_adapter_scale, 0.0]},
        }
        self.pipe.set_ip_adapter_scale(scale)

    # ── helpers ────────────────────────────────────────────────────────
    def _tensor_to_pil(self, t):
        """[1,3,H,W] float [0,1] → PIL Image."""
        arr = t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _pil_to_tensor(self, img):
        """PIL Image → [1,3,H,W] float [0,1]."""
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def _extract_canny(self, img):
        """PIL → canny edge map as PIL."""
        arr = np.array(img)
        edges = cv2.Canny(arr, self.cfg.canny_low, self.cfg.canny_high)
        edges = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges)

    # ── inference ──────────────────────────────────────────────────────
    @torch.no_grad()
    def __call__(self, content, style):
        """
        Parameters
        ----------
        content, style : Tensor [1, 3, H, W] in [0, 1]  OR  PIL.Image

        Returns
        -------
        Tensor [1, 3, H, W] in [0, 1]
        """
        # Convert tensors to PIL
        if isinstance(content, torch.Tensor):
            content_pil = self._tensor_to_pil(content)
        else:
            content_pil = content

        if isinstance(style, torch.Tensor):
            style_pil = self._tensor_to_pil(style)
        else:
            style_pil = style

        # Resize content to SDXL native size
        size = self.get_native_image_size()
        content_pil = content_pil.resize((size, size), Image.LANCZOS)
        style_pil = style_pil.resize((size, size), Image.LANCZOS)

        # Extract canny edges from content for ControlNet
        canny = self._extract_canny(content_pil)

        # Run pipeline
        result = self.pipe(
            prompt=self.cfg.prompt,
            negative_prompt=self.cfg.negative_prompt,
            ip_adapter_image=style_pil,
            image=canny,
            controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
        ).images[0]

        return self._pil_to_tensor(result).to(
            content.device if isinstance(content, torch.Tensor) else self.device
        )
