"""
StyleAligned – Shared Attention handler & DDIM inversion utilities.

Based on: https://github.com/google/style-aligned  (Apache-2.0)
Paper:  "Style Aligned Image Generation via Shared Attention" (2023)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from diffusers import StableDiffusionXLPipeline
from diffusers.models import attention_processor
from tqdm import tqdm

T = torch.Tensor


# ═══════════════════════════════════════════════════════════════════
#  Shared Attention primitives
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    shared_score_scale: float = 1.0
    shared_score_shift: float = 0.0
    only_self_level: float = 0.0


def expand_first(feat: T, scale=1.0) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    return torch.cat((feat, expand_first(feat, scale=scale)), dim=dim)


def _calc_mean_std(feat, eps=1e-5):
    return feat.mean(dim=-2, keepdims=True), (feat.var(dim=-2, keepdims=True) + eps).sqrt()


def adain(feat: T) -> T:
    mean, std = _calc_mean_std(feat)
    style_mean = expand_first(mean)
    style_std = expand_first(std)
    return (feat - mean) / std * style_std + style_mean


class DefaultAttentionProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = attention_processor.AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        return self.processor(attn, hidden_states, encoder_hidden_states, attention_mask)


class SharedAttentionProcessor(DefaultAttentionProcessor):
    def __init__(self, args: StyleAlignedArgs):
        super().__init__()
        self.share_attention = args.share_attention
        self.adain_queries = args.adain_queries
        self.adain_keys = args.adain_keys
        self.adain_values = args.adain_values
        self.full_attention_share = args.full_attention_share
        self.shared_score_scale = args.shared_score_scale
        self.shared_score_shift = args.shared_score_shift

    def shared_call(self, attn, hidden_states, encoder_hidden_states=None,
                    attention_mask=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, seq_len, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        head_dim = key.shape[-1] // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.adain_queries:
            query = adain(query)
        if self.adain_keys:
            key = adain(key)
        if self.adain_values:
            value = adain(value)
        if self.share_attention:
            key = concat_first(key, -2, scale=self.shared_score_scale)
            value = concat_first(value, -2)

        hidden_states = nnf.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        if self.full_attention_share:
            import einops
            b, n, d = hidden_states.shape
            hidden_states = einops.rearrange(hidden_states, '(k b) n d -> k (b n) d', k=2)
            hidden_states = super().__call__(attn, hidden_states,
                                             encoder_hidden_states=encoder_hidden_states,
                                             attention_mask=attention_mask, **kwargs)
            hidden_states = einops.rearrange(hidden_states, 'k (b n) d -> (k b) n d', n=n)
        else:
            hidden_states = self.shared_call(attn, hidden_states, hidden_states,
                                             attention_mask, **kwargs)
        return hidden_states


# ═══════════════════════════════════════════════════════════════════
#  Handler – register / unregister shared attention on a pipeline
# ═══════════════════════════════════════════════════════════════════

def _get_switch_vec(total, level):
    if level == 0:
        return torch.zeros(total, dtype=torch.bool)
    if level == 1:
        return torch.ones(total, dtype=torch.bool)
    to_flip = level > 0.5
    if to_flip:
        level = 1 - level
    num_switch = int(level * total)
    vec = torch.arange(total) % (total // max(num_switch, 1)) == 0
    return ~vec if to_flip else vec


def init_attention_processors(pipeline, args=None):
    unet = pipeline.unet
    num_self = len([n for n in unet.attn_processors if 'attn1' in n])
    only_self_vec = _get_switch_vec(num_self, 1 if args is None else args.only_self_level)
    procs = {}
    idx = 0
    for name in unet.attn_processors:
        if 'attn1' in name:
            if args is None or only_self_vec[idx // 2]:
                procs[name] = DefaultAttentionProcessor()
            else:
                procs[name] = SharedAttentionProcessor(args)
            idx += 1
        else:
            procs[name] = DefaultAttentionProcessor()
    unet.set_attn_processor(procs)


def register_shared_norm(pipeline, share_group_norm=True, share_layer_norm=True):
    def register(layer):
        if not hasattr(layer, 'orig_forward'):
            layer.orig_forward = layer.forward
        orig = layer.orig_forward

        def fwd(x):
            n = x.shape[-2]
            x = concat_first(x, dim=-2)
            x = orig(x)
            return x[..., :n, :]
        layer.forward = fwd
        return layer

    layers = []

    def collect(mod):
        if isinstance(mod, nn.LayerNorm) and share_layer_norm:
            layers.append(mod)
        elif isinstance(mod, nn.GroupNorm) and share_group_norm:
            layers.append(mod)
        else:
            for c in mod.children():
                collect(c)

    collect(pipeline.unet)
    return [register(l) for l in layers]


class Handler:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.norm_layers = []

    def register(self, args: StyleAlignedArgs):
        self.norm_layers = register_shared_norm(
            self.pipeline, args.share_group_norm, args.share_layer_norm)
        init_attention_processors(self.pipeline, args)

    def remove(self):
        for layer in self.norm_layers:
            layer.forward = layer.orig_forward
        self.norm_layers = []
        init_attention_processors(self.pipeline, None)


# ═══════════════════════════════════════════════════════════════════
#  DDIM Inversion
# ═══════════════════════════════════════════════════════════════════

def _get_text_emb(prompt, tokenizer, text_encoder, device):
    ids = tokenizer(prompt, padding='max_length',
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors='pt').input_ids
    with torch.no_grad():
        out = text_encoder(ids.to(device), output_hidden_states=True)
    pooled = out[0]
    hidden = out.hidden_states[-2]
    if prompt == '':
        return torch.zeros_like(hidden), torch.zeros_like(pooled)
    return hidden, pooled


def encode_text_sdxl(model, prompt):
    device = model._execution_device
    e1, p1 = _get_text_emb(prompt, model.tokenizer, model.text_encoder, device)
    e2, p2 = _get_text_emb(prompt, model.tokenizer_2, model.text_encoder_2, device)
    emb = torch.cat((e1, e2), dim=-1)
    proj_dim = model.text_encoder_2.config.projection_dim
    time_ids = model._get_add_time_ids(
        (1024, 1024), (0, 0), (1024, 1024), torch.float16, proj_dim).to(device)
    return {"text_embeds": p2, "time_ids": time_ids}, emb


def encode_text_sdxl_with_negative(model, prompt):
    cond_kw, emb = encode_text_sdxl(model, prompt)
    uncond_kw, emb_u = encode_text_sdxl(model, "")
    emb = torch.cat((emb_u, emb))
    merged = {
        "text_embeds": torch.cat((uncond_kw["text_embeds"], cond_kw["text_embeds"])),
        "time_ids": torch.cat((uncond_kw["time_ids"], cond_kw["time_ids"])),
    }
    return merged, emb


def encode_image(model, image_np):
    """Encode H×W×3 uint8 numpy → latent."""
    model.vae.to(dtype=torch.float32)
    img = torch.from_numpy(image_np).float() / 255.0
    img = (img * 2 - 1).permute(2, 0, 1).unsqueeze(0)
    z = model.vae.encode(img.to(model.vae.device))['latent_dist'].mean
    z = z * model.vae.config.scaling_factor
    model.vae.to(dtype=torch.float16)
    return z


def _next_step(model, noise, t, sample):
    s = model.scheduler
    dt = s.config.num_train_timesteps // s.num_inference_steps
    prev_t = min(t - dt, 999)
    a = s.alphas_cumprod[int(prev_t)] if prev_t >= 0 else s.final_alpha_cumprod
    a_next = s.alphas_cumprod[int(t)]
    x0 = (sample - (1 - a) ** 0.5 * noise) / a ** 0.5
    return a_next ** 0.5 * x0 + (1 - a_next) ** 0.5 * noise


@torch.no_grad()
def ddim_inversion(model, x0_np, prompt="", num_steps=50, guidance_scale=1.0):
    """DDIM-invert an image (H×W×3 uint8 numpy) → sequence of latents."""
    z0 = encode_image(model, x0_np)
    model.scheduler.set_timesteps(num_steps, device=z0.device)
    cond, emb = encode_text_sdxl_with_negative(model, prompt)
    z = z0.clone().half()
    all_z = [z0]
    for i in range(num_steps):
        t = model.scheduler.timesteps[num_steps - i - 1]
        z_in = torch.cat([z, z])
        noise = model.unet(z_in, t, encoder_hidden_states=emb,
                           added_cond_kwargs=cond)["sample"]
        nu, nc = noise.chunk(2)
        noise = nu + guidance_scale * (nc - nu)
        z = _next_step(model, noise, t, z)
        all_z.append(z)
    return torch.cat(all_z).flip(0)


def make_inversion_callback(zts, offset=0):
    def cb(pipeline, i, t, kwargs):
        kwargs['latents'][0] = zts[max(offset + 1, i + 1)].to(
            kwargs['latents'].device, kwargs['latents'].dtype)
        return kwargs
    return zts[offset], cb
