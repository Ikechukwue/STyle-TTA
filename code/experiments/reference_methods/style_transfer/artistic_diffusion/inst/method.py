"""
InST – Inversion-Based Style Transfer with Diffusion Models (CVPR 2023).

Uses Stable Diffusion 1.5 with textual inversion to capture style from a single
reference image, then applies the learned style embedding during DDIM denoising
of content latent.

Paper : arXiv 2211.13203 (CVPR 2023)
Repo  : https://github.com/zyxElsa/InST

Required:
  - runwayml/stable-diffusion-v1-5 (auto-downloaded from HF)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from types import SimpleNamespace
from tqdm import tqdm

from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


class Method:
    """InST: learn a style embedding via textual inversion, then transfer."""

    def __init__(self, device="cuda"):
        """
        Parameters
        ----------
        device : str
        """
        self.device = device
        self.cfg = self.get_default_config()
        self.style_embedding = None
        self._initialize_network()

    @staticmethod
    def get_default_config():
        return SimpleNamespace(
            base_model="runwayml/stable-diffusion-v1-5",
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=0.7,
            # Textual inversion params (for on-the-fly learning)
            ti_steps=500,
            ti_lr=5e-3,
            placeholder_token="<style>",
            initializer_token="painting",
        )

    @staticmethod
    def get_native_image_size():
        return 512

    # ── setup ──────────────────────────────────────────────────────────
    def _initialize_network(self):
        cfg = self.cfg
        self.scheduler = DDIMScheduler.from_pretrained(
            cfg.base_model, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.base_model, subfolder="text_encoder",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            cfg.base_model, subfolder="vae",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.base_model, subfolder="unet",
            torch_dtype=torch.float16,
        ).to(self.device)

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

    def _load_style_embedding(self, path):
        """Load a pre-computed textual inversion embedding."""
        data = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(data, dict):
            # Support InST format: {'string_to_param': {'*': tensor}}
            if 'string_to_param' in data:
                self.style_embedding = list(data['string_to_param'].values())[0]
            elif 'embedding' in data:
                self.style_embedding = data['embedding']
            else:
                self.style_embedding = list(data.values())[0]
        else:
            self.style_embedding = data
        self.style_embedding = self.style_embedding.to(self.device, torch.float16)

    # ── DDIM inversion ─────────────────────────────────────────────────
    @torch.no_grad()
    def _encode_image(self, image_tensor):
        """[1,3,H,W] float [0,1] → latent."""
        x = image_tensor.to(self.device, torch.float16) * 2 - 1
        z = self.vae.encode(x).latent_dist.mean
        return z * self.vae.config.scaling_factor

    @torch.no_grad()
    def _ddim_inversion(self, latent, text_emb, steps=50):
        """Invert latent to noise via DDIM."""
        self.scheduler.set_timesteps(steps)
        z = latent.clone()
        for i in range(len(self.scheduler.timesteps)):
            t = self.scheduler.timesteps[steps - i - 1]
            noise_pred = self.unet(z, t, encoder_hidden_states=text_emb).sample
            # DDIM forward step (inversion)
            alpha_t = self.scheduler.alphas_cumprod[t]
            dt = self.scheduler.config.num_train_timesteps // steps
            next_t = min(t + dt, self.scheduler.config.num_train_timesteps - 1)
            alpha_next = self.scheduler.alphas_cumprod[next_t]
            x0_pred = (z - (1 - alpha_t) ** 0.5 * noise_pred) / alpha_t ** 0.5
            z = alpha_next ** 0.5 * x0_pred + (1 - alpha_next) ** 0.5 * noise_pred
        return z

    # ── quick textual inversion ────────────────────────────────────────
    def _learn_style_embedding(self, style_tensor):
        """Learn a textual embedding from the style image (fast TI)."""
        cfg = self.cfg
        # Add placeholder token
        num_added = self.tokenizer.add_tokens([cfg.placeholder_token])
        placeholder_id = self.tokenizer.convert_tokens_to_ids(cfg.placeholder_token)
        init_id = self.tokenizer.encode(cfg.initializer_token, add_special_tokens=False)[0]

        # Resize token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            self.text_encoder.get_input_embeddings().weight[placeholder_id] = \
                self.text_encoder.get_input_embeddings().weight[init_id].clone()

        # Cast text encoder to float32 for TI training (float16 leaf tensors don't support autograd)
        self.text_encoder.float()

        # Prepare training: enable grad only on embedding weight
        emb_layer = self.text_encoder.get_input_embeddings()
        orig_weight = emb_layer.weight.data.clone()
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        emb_layer.weight.requires_grad_(True)
        optimizer = torch.optim.Adam([emb_layer.weight], lr=cfg.ti_lr)

        # Encode style image
        with torch.no_grad():
            z_style = self._encode_image(style_tensor)

        prompt = f"a painting in the style of {cfg.placeholder_token}"
        ids = self.tokenizer(prompt, padding="max_length",
                             max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)

        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        self.text_encoder.train()

        with torch.enable_grad():  # override outer torch.no_grad() from evaluation framework
            for step in tqdm(range(cfg.ti_steps), desc="Learning style embedding"):
                # Random timestep
                t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                                  (1,), device=self.device).long()
                noise = torch.randn_like(z_style)
                z_noisy = self.scheduler.add_noise(z_style, noise, t)

                # Get text conditioning (grad flows through emb_layer.weight)
                encoder_output = self.text_encoder(ids)[0].to(torch.float16)

                # Predict noise
                noise_pred = self.unet(z_noisy, t, encoder_hidden_states=encoder_output).sample

                loss = F.mse_loss(noise_pred.float(), noise.float())
                loss.backward()

                # Zero out gradients for all non-placeholder tokens, keep only placeholder
                with torch.no_grad():
                    if emb_layer.weight.grad is not None:
                        grad_placeholder = emb_layer.weight.grad[placeholder_id].clone()
                        emb_layer.weight.grad.zero_()
                        emb_layer.weight.grad[placeholder_id] = grad_placeholder
                optimizer.step()
                optimizer.zero_grad()

        self.text_encoder.eval()
        self.style_embedding = emb_layer.weight[placeholder_id:placeholder_id + 1].detach().to(torch.float16)

        # Restore original embeddings except placeholder
        emb_layer.weight.requires_grad_(False)
        with torch.no_grad():
            emb_layer.weight.data = orig_weight
            emb_layer.weight[placeholder_id] = self.style_embedding[0]

        # Cast back to float16 for inference
        self.text_encoder.half()

    # ── text encoding ──────────────────────────────────────────────────
    @torch.no_grad()
    def _get_text_embeddings(self, prompt):
        ids = self.tokenizer(prompt, padding="max_length",
                             max_length=self.tokenizer.model_max_length,
                             truncation=True, return_tensors="pt").input_ids.to(self.device)
        return self.text_encoder(ids)[0].to(torch.float16)

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _tensor_to_pil(t):
        arr = t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        return Image.fromarray((arr * 255).astype(np.uint8))

    @staticmethod
    def _pil_to_tensor(img):
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    # ── inference ──────────────────────────────────────────────────────
    def __call__(self, content, style):
        """
        Parameters
        ----------
        content, style : Tensor [1,3,H,W] in [0,1]

        Returns
        -------
        Tensor [1,3,H,W] in [0,1]
        """
        # Reset per-call: style embedding must be re-learned for each new style image.
        # Without this, all calls after the first reuse the first style's embedding.
        self.style_embedding = None

        cfg = self.cfg
        size = self.get_native_image_size()

        # Resize to 512
        c = F.interpolate(content.to(self.device), size=(size, size),
                          mode='bilinear', align_corners=False)
        s = F.interpolate(style.to(self.device), size=(size, size),
                          mode='bilinear', align_corners=False)

        # Learn style embedding on-the-fly if not pre-loaded (requires grad)
        if self.style_embedding is None:
            self._learn_style_embedding(s)

        with torch.no_grad():
            # Encode text with style embedding
            prompt = f"a painting in the style of {cfg.placeholder_token}"
            style_emb = self._get_text_embeddings(prompt)
            uncond_emb = self._get_text_embeddings("")

            # DDIM-invert content image
            z_content = self._encode_image(c)
            z_T = self._ddim_inversion(z_content, uncond_emb, cfg.num_inference_steps)

            # Denoise with style conditioning
            self.scheduler.set_timesteps(cfg.num_inference_steps)
            # Apply strength: skip first (1-strength) of steps
            start_step = int(cfg.num_inference_steps * (1 - cfg.strength))
            timesteps = self.scheduler.timesteps[start_step:]

            # Interpolate between inverted content and noise based on strength
            z = z_T.clone()
            for t in tqdm(timesteps, desc="Denoising"):
                z_in = torch.cat([z, z])
                t_batch = t.unsqueeze(0).to(self.device)
                emb = torch.cat([uncond_emb, style_emb])
                noise_pred = self.unet(z_in, t_batch, encoder_hidden_states=emb).sample
                noise_u, noise_c = noise_pred.chunk(2)
                noise_pred = noise_u + cfg.guidance_scale * (noise_c - noise_u)
                z = self.scheduler.step(noise_pred, t, z).prev_sample

            # Decode
            z = z / self.vae.config.scaling_factor
            img = self.vae.decode(z).sample
            img = (img + 1) / 2
            return img.clamp(0, 1).to(content.device)
