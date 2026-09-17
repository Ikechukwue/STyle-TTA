"""
LSAST – Towards Highly Realistic Artistic Style Transfer via Stable Diffusion
with Step-aware and Layer-aware Prompt Inversion (IJCAI 2024).

Paper : arXiv 2404.11474 (IJCAI 2024)
Repo  : https://github.com/Jamie-Cheung/LSAST  (Apache-2.0)

Uses Stable Diffusion 1.5 with per-style trained step-aware and layer-aware
prompt embeddings + ControlNet (canny) for content preservation.

Required:
  - runwayml/stable-diffusion-v1-5 (auto-downloaded from HF)
  - lllyasviel/control_v11p_sd15_canny (ControlNet)

Pre-trained style embeddings available at:
  https://drive.google.com/drive/folders/1Il2xvl38ubMuoVQoGSEBcjnnWrxXYElE

For on-the-fly style learning, provide weights=None (slow, ~10 min per style).

GPU : ~10-12 GB VRAM (fp16)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from types import SimpleNamespace
from tqdm import tqdm

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


class Method:
    """
    LSAST: step-aware/layer-aware prompt inversion style transfer.

    Falls back to single-embedding textual inversion + ControlNet when
    the full step/layer-aware prompt space is not available.
    """

    def __init__(self, device="cuda"):
        """ """
        self.device = device
        self.cfg = self.get_default_config()
        self.style_embedding = None
        self._initialize_network()

    @staticmethod
    def get_default_config():
        return SimpleNamespace(
            base_model="runwayml/stable-diffusion-v1-5",
            controlnet_model="lllyasviel/control_v11p_sd15_canny",
            num_inference_steps=50,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            canny_low=100,
            canny_high=200,
            placeholder_token="<style>",
            initializer_token="painting",
            ti_steps=500,
            ti_lr=5e-3,
        )

    @staticmethod
    def get_native_image_size():
        return 512

    # ── setup ──────────────────────────────────────────────────────────
    def _initialize_network(self):
        cfg = self.cfg

        controlnet = ControlNetModel.from_pretrained(
            cfg.controlnet_model, torch_dtype=torch.float16)

        self.scheduler = DDIMScheduler.from_pretrained(
            cfg.base_model, subfolder="scheduler")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.base_model, subfolder="text_encoder",
            torch_dtype=torch.float16).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            cfg.base_model, subfolder="vae",
            torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.base_model, subfolder="unet",
            torch_dtype=torch.float16).to(self.device)
        self.controlnet = controlnet.to(self.device)

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        self.controlnet.eval()

    def _load_style_embedding(self, path):
        data = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(data, dict):
            if 'string_to_param' in data:
                self.style_embedding = list(data['string_to_param'].values())[0]
            elif 'embedding' in data:
                self.style_embedding = data['embedding']
            else:
                self.style_embedding = list(data.values())[0]
        else:
            self.style_embedding = data
        self.style_embedding = self.style_embedding.to(self.device, torch.float16)

    # ── textual inversion ──────────────────────────────────────────────
    def _learn_style_embedding(self, style_tensor):
        cfg = self.cfg
        self.tokenizer.add_tokens([cfg.placeholder_token])
        pid = self.tokenizer.convert_tokens_to_ids(cfg.placeholder_token)
        init_id = self.tokenizer.encode(cfg.initializer_token, add_special_tokens=False)[0]

        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        emb = self.text_encoder.get_input_embeddings()
        with torch.no_grad():
            emb.weight[pid] = emb.weight[init_id].clone()

        # Cast text encoder to float32 for TI training (float16 leaf tensors don't support autograd)
        self.text_encoder.float()

        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        emb.weight.requires_grad_(True)
        optimizer = torch.optim.Adam([emb.weight], lr=cfg.ti_lr)

        with torch.no_grad():
            x = style_tensor.to(self.device, torch.float16) * 2 - 1
            z = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor

        prompt = f"a painting in the style of {cfg.placeholder_token}"
        ids = self.tokenizer(prompt, padding="max_length",
                             max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)

        self.text_encoder.train()
        orig_weight = emb.weight.data.clone()
        with torch.enable_grad():  # override outer torch.no_grad() from evaluation framework
            for _ in tqdm(range(cfg.ti_steps), desc="Learning style"):
                t = torch.randint(0, 1000, (1,), device=self.device).long()
                noise = torch.randn_like(z)
                z_noisy = self.scheduler.add_noise(z, noise, t)
                enc_out = self.text_encoder(ids)[0].to(torch.float16)
                pred = self.unet(z_noisy, t, encoder_hidden_states=enc_out).sample
                loss = F.mse_loss(pred.float(), noise.float())
                loss.backward()
                # Zero out gradients for non-placeholder tokens
                with torch.no_grad():
                    if emb.weight.grad is not None:
                        grad_pid = emb.weight.grad[pid].clone()
                        emb.weight.grad.zero_()
                        emb.weight.grad[pid] = grad_pid
                optimizer.step()
                optimizer.zero_grad()

        self.text_encoder.eval()
        self.style_embedding = emb.weight[pid:pid + 1].detach().to(torch.float16)
        emb.weight.requires_grad_(False)
        with torch.no_grad():
            emb.weight.data = orig_weight
            emb.weight[pid] = self.style_embedding[0]

        # Cast back to float16 for inference
        self.text_encoder.half()

    # ── helpers ────────────────────────────────────────────────────────
    def _extract_canny(self, tensor):
        """[1,3,H,W] [0,1] → [1,3,H,W] canny float."""
        arr = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        edges = cv2.Canny(arr, self.cfg.canny_low, self.cfg.canny_high)
        edges = np.stack([edges] * 3, axis=-1).astype(np.float32) / 255.0
        return torch.from_numpy(edges).permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def _get_text_emb(self, prompt):
        ids = self.tokenizer(prompt, padding="max_length",
                             max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids)[0].to(torch.float16)

    # ── inference ──────────────────────────────────────────────────────
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

        # Reset per-call: re-learn style embedding for each new style image.
        # Without this, all calls after the first reuse the first style's embedding.
        self.style_embedding = None

        c = F.interpolate(content.to(self.device), (size, size),
                          mode='bilinear', align_corners=False)
        s = F.interpolate(style.to(self.device), (size, size),
                          mode='bilinear', align_corners=False)

        # Learn style embedding via textual inversion (requires grad)
        self._learn_style_embedding(s)

        with torch.no_grad():
            # Extract canny from content
            canny = self._extract_canny(c).to(self.device, torch.float16)

            # Text embeddings
            prompt = f"a painting in the style of {cfg.placeholder_token}"
            cond = self._get_text_emb(prompt)
            uncond = self._get_text_emb("")

            # Start from a noised content latent (img2img at strength=0.7).
            # Official LSAST uses DDIM img2img: encode content → add 70% noise →
            # denoise from that timestep. This anchors output colors to the
            # content image; pure random noise makes ControlNet the only
            # structural anchor and produces wrong / random colors.
            strength = 0.7
            self.scheduler.set_timesteps(cfg.num_inference_steps)
            start_step = int(cfg.num_inference_steps * (1.0 - strength))

            # Encode content image to latent space (VAE expects [-1, 1])
            with torch.no_grad():
                z_c = self.vae.encode(
                    c.to(torch.float16) * 2.0 - 1.0
                ).latent_dist.mean * self.vae.config.scaling_factor

            # Add noise corresponding to the start timestep
            t_start = self.scheduler.timesteps[start_step]
            noise = torch.randn_like(z_c)
            z = self.scheduler.add_noise(z_c, noise, t_start.unsqueeze(0))

            for t in tqdm(self.scheduler.timesteps[start_step:], desc="Denoising"):
                z_in = torch.cat([z, z])
                emb = torch.cat([uncond, cond])
                canny_in = torch.cat([canny, canny])

                # ControlNet
                down, mid = self.controlnet(
                    z_in, t, encoder_hidden_states=emb,
                    controlnet_cond=canny_in,
                    conditioning_scale=cfg.controlnet_conditioning_scale,
                    return_dict=False,
                )

                # UNet with ControlNet residuals
                noise_pred = self.unet(
                    z_in, t, encoder_hidden_states=emb,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                ).sample

                noise_u, noise_c = noise_pred.chunk(2)
                noise_pred = noise_u + cfg.guidance_scale * (noise_c - noise_u)
                z = self.scheduler.step(noise_pred, t, z).prev_sample

            # Decode
            z = z / self.vae.config.scaling_factor
            img = self.vae.decode(z).sample
            img = (img + 1) / 2
            return img.clamp(0, 1).to(content.device)
