"""VeloEdit: Velocity Field Intervention for Style Transfer using FLUX.1-Kontext-dev.

Implements the official VeloEdit approach (https://github.com/xmulzq/VeloEdit) adapted
for style transfer. Uses FLUX.1-Kontext-dev — a flow-matching image-editing model —
with a custom deterministic Euler sampling loop and element-wise velocity field
intervention.

Style transfer protocol:
  - Content image is passed as the Kontext conditioning reference (structure to preserve).
  - A BLIP-2 caption of the style image is used as the editing prompt.
  - At each integration step the transformer-predicted velocity v_pred is compared
    element-wise against the reference velocity v_ref = (z_t - z_content) / sigma.
    Where |v_pred - v_ref| is small (content-consistent), v_ref replaces v_pred to
    enforce structural fidelity. Where the deviation is large (style-driven), v_pred
    is kept, allowing stylistic changes to propagate.

Reference: VeloEdit (2025) — Velocity Field Analysis and Intervention for Image Editing
"""

import numpy as np
import torch
from PIL import Image
from typing import Optional


# ---------------------------------------------------------------------------
# Velocity-field helpers (mirror of official VeloEdit core/intervention.py)
# ---------------------------------------------------------------------------

def _element_similarity(v_pred: torch.Tensor, v_ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = torch.abs(v_pred.float() - v_ref.float())
    ref_abs = torch.abs(v_ref.float()) + eps
    return ref_abs / (ref_abs + diff)


def _apply_intervention(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    threshold: float,
    blend: bool = False,
    blend_weight: float = 0.5,
) -> torch.Tensor:
    dtype = v_pred.dtype
    sim = _element_similarity(v_pred, v_ref)
    v_ref_d = v_ref.to(dtype)
    result = torch.where(sim >= threshold, v_ref_d, v_pred)
    if blend:
        blended = blend_weight * v_ref_d + (1.0 - blend_weight) * v_pred
        result = torch.where(sim < threshold, blended, result)
    return result


def _euler_step(z: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    return z + (sigma_next - sigma) * v


# ---------------------------------------------------------------------------
# Main Method class
# ---------------------------------------------------------------------------

class Method:
    """VeloEdit style transfer via FLUX.1-Kontext-dev velocity field intervention."""

    def __init__(
        self,
        pretrained_weights: Optional[str] = None,
        device: str = "cpu",
        model_id: str = "black-forest-labs/FLUX.1-Kontext-dev",
        num_inference_steps: int = 28,
        guidance_scale: float = 2.5,
        similarity_threshold: float = 0.8,
        intervention_steps: int = 14,
        enable_blend: bool = False,
        blend_weight: float = 0.5,
    ):
        self.device = device
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.similarity_threshold = similarity_threshold
        self.intervention_steps = intervention_steps
        self.enable_blend = enable_blend
        self.blend_weight = blend_weight
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
    # Core VeloEdit sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_veloedit(
        self,
        content_pil: Image.Image,
        prompt: str,
        height: int,
        width: int,
    ) -> Image.Image:
        """Run one VeloEdit pass: custom Euler loop with velocity intervention."""
        pipe = self.pipe
        device = pipe.device
        transformer_dtype = pipe.transformer.dtype

        # --- Text embeddings ---
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

        # --- Prepare content latents (used as Kontext conditioning) ---
        image_input = pipe.image_processor.preprocess(content_pil, height=height, width=width)
        image_input = image_input.to(device=device, dtype=torch.float32)
        num_channels_latents = pipe.transformer.config.in_channels // 4
        generator = torch.Generator(device=device).manual_seed(42)
        latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
            image_input,
            1,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            None,
        )

        # Combine latent and image positional ids for the Kontext transformer
        if image_ids is not None:
            combined_ids = torch.cat([latent_ids, image_ids], dim=0)
        else:
            combined_ids = latent_ids

        # Cast to transformer dtype
        latents = latents.to(transformer_dtype)
        image_latents = image_latents.to(transformer_dtype) if image_latents is not None else None

        # Reference latent: encoded content image (target for content preservation)
        reference_latent = image_latents.clone() if image_latents is not None else latents.clone()

        # --- Build guidance tensor ---
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], self.guidance_scale, device=device, dtype=torch.float32
            )
        else:
            guidance = None

        # --- Sigma schedule (mirrors official VeloEdit) ---
        from diffusers.pipelines.flux.pipeline_flux_kontext import calculate_shift, retrieve_timesteps
        num_steps = self.num_inference_steps
        sigmas_input = np.linspace(1.0, 1.0 / num_steps, num_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        _, _ = retrieve_timesteps(pipe.scheduler, num_steps, device, sigmas=sigmas_input, mu=mu)
        sigma_schedule = pipe.scheduler.sigmas.float()  # length = num_steps + 1

        # --- Velocity prediction closure ---
        def v_pred_fn(z: torch.Tensor, sigma: float) -> torch.Tensor:
            model_in = torch.cat([z, image_latents], dim=1) if image_latents is not None else z
            t = torch.full([model_in.shape[0]], sigma, device=device, dtype=torch.float32)
            out = pipe.transformer(
                hidden_states=model_in,
                timestep=t,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            # Trim output back to the noise-latent size (discard the image-conditioning part)
            return out[:, : latents.size(1)]

        # --- Euler integration with velocity intervention ---
        z = latents
        for i in range(num_steps):
            sigma = sigma_schedule[i].item()
            sigma_next = sigma_schedule[i + 1].item()

            v_pred = v_pred_fn(z, sigma)

            if i < self.intervention_steps:
                # Reference velocity: direction from z_t back toward encoded content
                v_ref = (z - reference_latent) / (sigma + 1e-8)
                v_ref = v_ref.to(v_pred.dtype)
                v_pred = _apply_intervention(
                    v_pred, v_ref, self.similarity_threshold,
                    self.enable_blend, self.blend_weight,
                )

            z = _euler_step(z, v_pred, sigma, sigma_next).to(transformer_dtype)

        # --- Decode ---
        z_unpacked = pipe._unpack_latents(z, height, width, pipe.vae_scale_factor)
        z_vae = (z_unpacked / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(z_vae.to(pipe.vae.dtype), return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

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
        """Apply VeloEdit style transfer.

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
            out_pil = self._run_veloedit(c_pil, prompt, size, size)
            results.append(self._pil_to_tensor(out_pil).squeeze(0).float())

        return torch.stack(results).clamp(0, 1).to(content.device)
