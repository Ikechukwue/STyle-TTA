"""Diffusion-based style transfer with attention overload and KV-cache support."""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
KVCache = Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]
FeatureCache = Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]


class StyleInjectionDiffusion:
    """Training-free diffusion style injection with overloaded attention.

    The U-Net attention mechanism is **overloaded**: it can process both full
    reference images (encoding them to extract features) *and* directly accept
    pre-computed K/V tensors.  The test image **always** provides the content
    Query (Q).

    Parameters
    ----------
    sd_version : str
        Stable Diffusion version id (``'1.4'``, ``'1.5'``).
    gamma : float
        Query preservation weight.  ``Q_mix = gamma * Q_content + (1-gamma) * Q_current``.
    temperature : float
        Attention temperature scaling factor.
    ddim_steps : int
        Number of DDIM sampling / inversion steps.
    injection_layers : list[int]
        UNet decoder layer indices to inject style into.
    use_adain : bool
        Apply AdaIN latent warm-start before denoising.
    start_step : int
        First timestep index at which injection is applied.
    device : torch.device | str | None
        Target device (auto-detected when ``None``).
    """

    _SD_MODEL_MAP = {
        "1.4": "CompVis/stable-diffusion-v1-4",
        "1.5": "runwayml/stable-diffusion-v1-5",
    }

    def __init__(
        self,
        sd_version: str = "1.4",
        gamma: float = 0.75,
        temperature: float = 1.5,
        ddim_steps: int = 50,
        injection_layers: Optional[List[int]] = None,
        use_adain: bool = True,
        start_step: int = 49,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.sd_version = sd_version
        self.gamma = gamma
        self.temperature = temperature
        self.ddim_steps = ddim_steps
        self.injection_layers = injection_layers or [6, 7, 8, 9, 10, 11]
        self.use_adain = use_adain
        self.start_step = start_step

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Lazy-loaded SD components
        self.pipe = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.text_encoder = None
        self.tokenizer = None
        self._uncond_emb: Optional[torch.Tensor] = None

        # Hook book-keeping
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._cur_t: Optional[int] = None

        self._is_initialised = False

    # ==================================================================
    # Initialisation
    # ==================================================================
    def initialise(self) -> "StyleInjectionDiffusion":
        """Download / load Stable Diffusion and prepare components."""
        if self._is_initialised:
            return self

        from diffusers import StableDiffusionPipeline, DDIMScheduler

        model_id = self._SD_MODEL_MAP.get(
            self.sd_version, "runwayml/stable-diffusion-v1-5"
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = self.scheduler
        self.scheduler.set_timesteps(self.ddim_steps)

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        self.vae.eval().requires_grad_(False)
        self.unet.eval().requires_grad_(False)

        # Pre-compute unconditional embedding (empty prompt)
        self._uncond_emb = self._text_embeddings("")

        self._is_initialised = True
        return self

    # ==================================================================
    # Public API — Overloaded transfer
    # ==================================================================
    @torch.no_grad()
    def transfer(
        self,
        content: torch.Tensor,
        *,
        style_image: Optional[torch.Tensor] = None,
        style_kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """Perform style injection.

        **Overloaded inputs** — provide *either* ``style_image`` or
        ``style_kv_cache``:

        * ``style_image``: ``(B, 3, H, W)`` in [0, 1].  The image is
          encoded and DDIM-inverted to extract K/V features.
        * ``style_kv_cache``: Pre-computed ``{layer -> {t -> (K, V)}}``
          dictionary, bypassing the encoder entirely.

        Parameters
        ----------
        content : Tensor
            Content (test) image ``(B, 3, H, W)`` in [0, 1].
        style_image : Tensor, optional
            Full reference image.
        style_kv_cache : KVCache, optional
            Pre-computed K/V tensors per layer per timestep.

        Returns
        -------
        Tensor
            Stylised image ``(B, 3, H, W)`` in [0, 1].
        """
        if not self._is_initialised:
            self.initialise()

        if style_image is None and style_kv_cache is None:
            raise ValueError("Provide either style_image or style_kv_cache.")

        content = content.to(device=self.device, dtype=self.dtype)

        # --- Encode content -------------------------------------------------
        content_norm = content * 2.0 - 1.0
        content_latent = self._encode(content_norm)
        content_inverted, content_features = self._ddim_inversion(
            content_latent, extract_features=True
        )

        # --- Get style K/V ---------------------------------------------------
        if style_kv_cache is not None:
            # Direct KV path — no encoding needed.
            style_kv = style_kv_cache
            # AdaIN still needs style latent statistics; approximate from the
            # V tensors' mean (or skip AdaIN when cache is used).
            style_inverted = content_inverted  # fallback
        else:
            # Full-image path — encode + invert
            style_image = style_image.to(device=self.device, dtype=self.dtype)
            style_norm = style_image * 2.0 - 1.0
            style_latent = self._encode(style_norm)
            style_inverted, style_features = self._ddim_inversion(
                style_latent, extract_features=True
            )
            # Build KV-only cache from full features
            style_kv = self._features_to_kv(style_features)

        # --- Build injection dict: content Q + style K/V ---------------------
        inject_dict = self._build_injection_dict(content_features, style_kv)

        # --- AdaIN warm-start ------------------------------------------------
        if self.use_adain and style_image is not None:
            initial_latent = self._latent_adain(content_inverted, style_inverted)
        else:
            initial_latent = content_inverted

        # --- Denoise with injection ------------------------------------------
        stylised_latent = self._ddim_sample_inject(initial_latent, inject_dict)

        # --- Decode ----------------------------------------------------------
        output = self._decode(stylised_latent)
        output = (output + 1.0) / 2.0
        return output.clamp(0, 1).to(content.dtype)

    # ==================================================================
    # KV-cache generation (offline)
    # ==================================================================
    @torch.no_grad()
    def build_kv_cache(self, image: torch.Tensor) -> KVCache:
        """Pre-compute K/V tensors for *image* (offline step).

        Parameters
        ----------
        image : Tensor
            Reference image ``(1, 3, H, W)`` in [0, 1].

        Returns
        -------
        KVCache
            ``{layer_name: {timestep: (K, V)}}``.
        """
        if not self._is_initialised:
            self.initialise()

        image = image.to(device=self.device, dtype=self.dtype)
        latent = self._encode(image * 2.0 - 1.0)
        _, features = self._ddim_inversion(latent, extract_features=True)
        return self._features_to_kv(features)

    # ==================================================================
    # Internal — VAE encode / decode
    # ==================================================================
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``(B,3,H,W)`` in [-1,1] → latent ``(B,4,h,w)``."""
        latents = self.vae.encode(images).latent_dist.mode()
        return latents * 0.18215

    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent → ``(B,3,H,W)`` in [-1,1]."""
        return self.vae.decode(latents / 0.18215).sample

    # ==================================================================
    # Internal — text embeddings
    # ==================================================================
    def _text_embeddings(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(
            [text],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(tokens.input_ids.to(self.text_encoder.device))[0]

    # ==================================================================
    # Internal — DDIM inversion
    # ==================================================================
    def _ddim_inversion(
        self,
        latent: torch.Tensor,
        extract_features: bool = True,
    ) -> Tuple[torch.Tensor, FeatureCache]:
        features: FeatureCache = {
            f"layer_{i}_attn": {} for i in self.injection_layers
        }

        if extract_features:
            self._register_extraction_hooks(features)

        current = latent.clone()
        timesteps_rev = list(reversed(self.scheduler.timesteps))
        n_steps = len(self.scheduler.timesteps)

        for i in range(n_steps):
            t = timesteps_rev[i]
            self._cur_t = t.item()

            noise_pred = self.unet(
                current, t, encoder_hidden_states=self._uncond_emb
            ).sample

            cur_t_val = max(0, t.item() - (1000 // n_steps))
            next_t_val = t.item()
            alpha_t = self.scheduler.alphas_cumprod[cur_t_val]
            alpha_next = self.scheduler.alphas_cumprod[next_t_val]

            current = (
                (current - (1 - alpha_t).sqrt() * noise_pred)
                * (alpha_next.sqrt() / alpha_t.sqrt())
                + (1 - alpha_next).sqrt() * noise_pred
            )

        self._remove_hooks()
        return current, features

    # ==================================================================
    # Internal — DDIM sampling with injection
    # ==================================================================
    def _ddim_sample_inject(
        self,
        noisy_latent: torch.Tensor,
        inject_dict: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ) -> torch.Tensor:
        self._register_injection_hooks(inject_dict)

        current = noisy_latent.clone()
        total = len(self.scheduler.timesteps)

        for i, t in enumerate(self.scheduler.timesteps):
            idx = total - i - 1
            self._cur_t = t.item() if idx < self.start_step else None

            noise_pred = self.unet(
                current, t, encoder_hidden_states=self._uncond_emb
            ).sample
            current = self.scheduler.step(noise_pred, t, current, return_dict=False)[0]

        self._remove_hooks()
        return current

    # ==================================================================
    # Internal — AdaIN in latent space
    # ==================================================================
    @staticmethod
    def _latent_adain(
        content_lat: torch.Tensor, style_lat: torch.Tensor
    ) -> torch.Tensor:
        c_mu = content_lat.mean(dim=[2, 3], keepdim=True)
        c_sig = content_lat.std(dim=[2, 3], keepdim=True) + 1e-4
        s_mu = style_lat.mean(dim=[2, 3], keepdim=True)
        s_sig = style_lat.std(dim=[2, 3], keepdim=True) + 1e-4
        return (content_lat - c_mu) / c_sig * s_sig + s_mu

    # ==================================================================
    # Internal — feature / cache helpers
    # ==================================================================
    @staticmethod
    def _features_to_kv(features: FeatureCache) -> KVCache:
        """Strip Q from a full feature cache, keeping only K and V."""
        kv: KVCache = {}
        for layer, timesteps in features.items():
            kv[layer] = {}
            for t, (q, k, v) in timesteps.items():
                kv[layer][t] = (k, v)
        return kv

    @staticmethod
    def _build_injection_dict(
        content_features: FeatureCache, style_kv: KVCache
    ) -> Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Combine content Q with style K/V into the injection dict."""
        inject: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        for layer in style_kv:
            inject[layer] = {}
            for t in style_kv[layer]:
                if layer in content_features and t in content_features[layer]:
                    q_c = content_features[layer][t][0]
                    k_s, v_s = style_kv[layer][t]
                    inject[layer][t] = (q_c, k_s, v_s)
        return inject

    # ==================================================================
    # Internal — attention layer utilities
    # ==================================================================
    def _get_attention_layers(self) -> List[Optional[nn.Module]]:
        """Return the 12 self-attention modules from the UNet decoder."""
        modules: List[Optional[nn.Module]] = []
        for i in range(12):
            blk_idx, layer_idx = divmod(i, 3)
            if blk_idx < len(self.unet.up_blocks):
                block = self.unet.up_blocks[blk_idx]
                if hasattr(block, "attentions") and layer_idx < len(block.attentions):
                    tb = block.attentions[layer_idx]
                    if hasattr(tb, "transformer_blocks"):
                        modules.append(tb.transformer_blocks[0].attn1)
                    else:
                        modules.append(None)
                else:
                    modules.append(None)
            else:
                modules.append(None)
        return modules

    # ------------------------------------------------------------------
    # Extraction hooks
    # ------------------------------------------------------------------
    def _register_extraction_hooks(self, features: FeatureCache) -> None:
        attn_layers = self._get_attention_layers()
        for idx in self.injection_layers:
            if idx >= len(attn_layers) or attn_layers[idx] is None:
                continue
            module = attn_layers[idx]
            name = f"layer_{idx}_attn"

            def _make(n: str):
                def hook(mod, inp, out):
                    hs = inp[0]
                    q = mod.head_to_batch_dim(mod.to_q(hs))
                    k = mod.head_to_batch_dim(mod.to_k(hs))
                    v = mod.head_to_batch_dim(mod.to_v(hs))
                    if self._cur_t is not None:
                        features[n][int(self._cur_t)] = (
                            q.detach().clone(),
                            k.detach().clone(),
                            v.detach().clone(),
                        )
                return hook

            self._hooks.append(module.register_forward_hook(_make(name)))

    # ------------------------------------------------------------------
    # Injection hooks (the overloaded SA forward pass)
    # ------------------------------------------------------------------
    def _register_injection_hooks(
        self,
        inject_dict: Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ) -> None:
        attn_layers = self._get_attention_layers()
        for idx in self.injection_layers:
            if idx >= len(attn_layers) or attn_layers[idx] is None:
                continue
            module = attn_layers[idx]
            name = f"layer_{idx}_attn"

            def _make(n: str):
                def hook(mod, inp, out):
                    if self._cur_t is None or int(self._cur_t) not in inject_dict.get(n, {}):
                        return out

                    hs = inp[0]
                    ndim = hs.ndim
                    if ndim == 4:
                        B, C, H, W = hs.shape
                        hs = hs.view(B, C, H * W).transpose(1, 2)

                    q_cur = mod.head_to_batch_dim(mod.to_q(hs))
                    q_c, k_s, v_s = inject_dict[n][int(self._cur_t)]

                    # Query mixing
                    q_hat = self.gamma * q_c + (1 - self.gamma) * q_cur
                    q_hat = q_hat * self.temperature

                    # Ensure shape compat
                    if k_s.shape[0] != q_hat.shape[0]:
                        k_s = k_s[: q_hat.shape[0]]
                        v_s = v_s[: q_hat.shape[0]]

                    scale = 1.0 / (q_hat.shape[-1] ** 0.5)
                    attn = F.softmax(
                        torch.bmm(q_hat, k_s.transpose(-1, -2)) * scale, dim=-1
                    )
                    hs_out = torch.bmm(attn, v_s)

                    hs_out = mod.batch_to_head_dim(hs_out)
                    hs_out = mod.to_out[0](hs_out)
                    hs_out = mod.to_out[1](hs_out)

                    if ndim == 4:
                        hs_out = hs_out.transpose(-1, -2).reshape(B, C, H, W)
                    if getattr(mod, "residual_connection", False):
                        hs_out = hs_out + inp[0]
                    if hasattr(mod, "rescale_output_factor"):
                        hs_out = hs_out / mod.rescale_output_factor
                    return hs_out

                return hook

            self._hooks.append(module.register_forward_hook(_make(name)))

    # ------------------------------------------------------------------
    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
