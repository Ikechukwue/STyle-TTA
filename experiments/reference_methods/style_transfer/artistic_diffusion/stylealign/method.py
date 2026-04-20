"""
StyleAligned – Style transfer via Shared Attention in SDXL.

Training-free method. Uses DDIM inversion of the style image as reference,
then generates from inverted content latent with shared attention ensuring
style consistency. ControlNet (depth) optionally preserves content structure.

Paper : "Style Aligned Image Generation via Shared Attention" (2023)
Repo  : https://github.com/google/style-aligned  (Apache-2.0)

Required HuggingFace models (auto-downloaded):
  - stabilityai/stable-diffusion-xl-base-1.0
  - (optional) diffusers/controlnet-depth-sdxl-1.0

GPU : ~16-24 GB VRAM (fp16)
"""

import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

from diffusers import StableDiffusionXLPipeline, DDIMScheduler

from experiments.reference_methods.style_transfer.artistic_diffusion.stylealign.net import (
    StyleAlignedArgs, Handler, ddim_inversion, make_inversion_callback,
)


class Method:
    """StyleAligned style transfer via shared self-attention in SDXL."""

    def __init__(self, device="cuda"):
        self.device = device
        self.cfg = self.get_default_config()
        self._initialize_network()

    @staticmethod
    def get_default_config():
        return SimpleNamespace(
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            num_inference_steps=50,
            guidance_scale=5.0,
            inv_guidance_scale=1.0,
            inversion_offset=5,
            prompt="",
        )

    @staticmethod
    def get_native_image_size():
        return 1024

    # ── setup ──────────────────────────────────────────────────────────
    def _initialize_network(self):
        cfg = self.cfg
        scheduler = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            clip_sample=False, set_alpha_to_one=False,
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg.base_model, scheduler=scheduler, torch_dtype=torch.float16,
        ).to(self.device)

        self.handler = Handler(self.pipe)
        self.sa_args = StyleAlignedArgs(
            share_group_norm=True,
            share_layer_norm=True,
            share_attention=True,
            adain_queries=True,
            adain_keys=True,
        )

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _tensor_to_np(t):
        """[1,3,H,W] float [0,1] → H×W×3 uint8 numpy."""
        arr = t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        return (arr * 255).astype(np.uint8)

    @staticmethod
    def _pil_to_tensor(img):
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # ── inference ──────────────────────────────────────────────────────
    @torch.no_grad()
    def __call__(self, content, style):
        """
        Parameters
        ----------
        content, style : Tensor [1, 3, H, W] in [0, 1]

        Returns
        -------
        Tensor [1, 3, H, W] in [0, 1]
        """
        cfg = self.cfg
        size = self.get_native_image_size()

        # Convert to numpy for DDIM inversion
        if isinstance(content, torch.Tensor):
            c_np = self._tensor_to_np(content)
        else:
            c_np = np.array(content)

        if isinstance(style, torch.Tensor):
            s_np = self._tensor_to_np(style)
        else:
            s_np = np.array(style)

        # Resize to SDXL native
        c_pil = Image.fromarray(c_np).resize((size, size), Image.LANCZOS)
        s_pil = Image.fromarray(s_np).resize((size, size), Image.LANCZOS)
        c_np = np.array(c_pil)
        s_np = np.array(s_pil)

        # DDIM-invert both images
        zts_style = ddim_inversion(
            self.pipe, s_np, prompt=cfg.prompt,
            num_steps=cfg.num_inference_steps,
            guidance_scale=cfg.inv_guidance_scale,
        )
        zts_content = ddim_inversion(
            self.pipe, c_np, prompt=cfg.prompt,
            num_steps=cfg.num_inference_steps,
            guidance_scale=cfg.inv_guidance_scale,
        )

        offset = cfg.inversion_offset
        zT_style, cb_style = make_inversion_callback(zts_style, offset=offset)
        zT_content, cb_content = make_inversion_callback(zts_content, offset=offset)

        # Stack latents: [style_ref, content_target]
        latents = torch.cat([zT_style, zT_content])

        # Register shared attention
        self.handler.register(self.sa_args)

        # Combined callback that injects correct latents for each batch item
        def combined_cb(pipeline, i, t, kwargs):
            lat = kwargs['latents']
            lat[0] = zts_style[max(offset + 1, i + 1)].to(lat.device, lat.dtype)
            lat[1] = zts_content[max(offset + 1, i + 1)].to(lat.device, lat.dtype)
            return kwargs

        # Run pipeline with 2-image batch (reference + target)
        prompts = [cfg.prompt, cfg.prompt]
        images = self.pipe(
            prompt=prompts,
            latents=latents,
            callback_on_step_end=combined_cb,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
        ).images

        # Remove shared attention
        self.handler.remove()

        # Return the second image (target / content with style)
        result = images[1]
        out = self._pil_to_tensor(result)
        return out.to(content.device if isinstance(content, torch.Tensor) else self.device)
