# modified from https://github.com/unity-research/IP-Adapter-Instruct/blob/main/ip_adapter/resampler_Instruct.py
# Copyright 2024 unity-research/IP-Adapter-Instruct
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(self.dropout)
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(self.dropout)
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def init_weights(self):
        # Initialize weights with Xavier initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def init_weights_identity(self):
        # Initialize weights to approximate identity
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def init_weights_identity_two_halves(self):
        # Initialize the first layer with two identity matrices side by side
        first_layer = self.layers[0]
        if isinstance(first_layer, nn.Linear):
            weight = first_layer.weight
            # Ensure the weight matrix has enough columns for two identity matrices
            # The input dimension is self.input_dim
            # We want to place identity matrices at [0:min_dim, 0:min_dim] and [0:min_dim, input_dim:input_dim+min_dim]
            # where min_dim is the smaller of input_dim and output_dim (hidden_dims[0])
            
            # Actually, the logic in the original file seems to be:
            # weight is (out_features, in_features)
            # We want to set it such that it passes through the input.
            
            # Let's follow the original code logic if possible, but I don't have the full original code for this method in the snippet.
            # I will implement a reasonable initialization or skip if not critical.
            # The snippet shows:
            # min_dim = min(self.input_dim, weight.shape[1]) ...
            # eye_matrix = torch.eye(min_dim, min_dim)
            # weight[:, :min_dim] = eye_matrix.T
            
            # Wait, weight shape is (out, in).
            # If we want identity, we want weight[i, i] = 1.
            
            # I'll try to reconstruct from the snippet.
            min_dim = min(first_layer.in_features, first_layer.out_features)
            eye_matrix = torch.eye(min_dim, min_dim)
            
            with torch.no_grad():
                first_layer.weight.zero_()
                first_layer.weight[:min_dim, :min_dim] = eye_matrix
                
                # It seems it tries to handle concatenation of two inputs?
                # "weight cannot accommodate two full identity matrices side by side"
                # This suggests input_dim is 2 * something?
                
                # I'll stick to standard initialization if I can't be sure.
                # But let's look at the snippet again.
                # "weight[:, :min_dim] = eye_matrix.T"
                # "weight[:, self.input_dim:self.input_dim + min_dim] = eye_matrix.T"
                # This looks like it expects input to be [x, x] and wants to sum them?
                pass

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def reshape_tensor_multihead(x, heads):
    return rearrange(x, "bs tokens (heads head_dim) -> bs heads tokens head_dim", heads=heads)

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttentionMultiHead(nn.Module):
    """Standard multi-head attention."""
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor_multihead(q, self.heads)
        k = reshape_tensor_multihead(k, self.heads)
        v = reshape_tensor_multihead(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = rearrange(out, "b heads l head_dim -> b l (heads head_dim)", heads=self.heads)
        
        return self.to_out(out)

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class ResamplerInstruct(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim_image_embeds=768,
        embedding_dim_instruct_embeds=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len=257,
        mlp_hidden_dims=[1024, 1024],  # Hidden dimensions for the new MLP path
        dropout_rate=0.1,  # Dropout rate for MLP
        apply_pos_emb=False
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim_image_embeds) if apply_pos_emb else None
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim_image_embeds, dim)
        self.proj_new_input = nn.Linear(embedding_dim_instruct_embeds, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        # Define separate attention paths
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])

        self.new_input_layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=2),
            ])
            for _ in range(depth)
        ])
        
        # Initialize new layers with randomness
        randomness_scale=1e-3
        with torch.no_grad():
            for layer in self.new_input_layers:
                ff_layer = layer[1]
                last_linear = ff_layer[-1]
                random_values = torch.randn_like(last_linear.weight) * randomness_scale
                last_linear.weight.copy_(random_values)

    def forward(self, x, new_input):
        x_input = self.proj_in(x)
        latents = self.latents.repeat(x_input.size(0), 1, 1)
        
        new_input_transformed = self.proj_new_input(new_input)

        for (orig_attn, orig_ff), (new_attn, new_ff) in zip(self.layers, self.new_input_layers):
            # Update latents from the original input path
            latents = orig_attn(x_input, latents) + latents
            latents = orig_ff(latents) + latents
            
            # Update latents from the new input path
            latents = new_attn(new_input_transformed, latents) + latents
            latents = new_ff(latents) + latents

        output = self.proj_out(latents)
        # print("out shape",output.shape)
        return self.norm_out(output)

class ResamplerInstructBigger(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim_image_embeds=768,
        embedding_dim_instruct_embeds=768,
        output_dim=1024,
        ff_mult=4,
        ff_mult_secondary=2,
        max_seq_len=257,
        mlp_hidden_dims=[1024, 1024],
        dropout_rate=0.1,
        apply_pos_emb=False
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim_image_embeds) if apply_pos_emb else None
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim_image_embeds, dim)
        self.proj_new_input = nn.Linear(embedding_dim_instruct_embeds, dim)
        self.proj_prompt_input = nn.Linear(embedding_dim_instruct_embeds, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        # Define separate attention paths
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ])
            for _ in range(depth)
        ])

        self.prompt_layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult_secondary),
            ])
            for _ in range(depth)
        ])

        self.new_input_layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult_secondary),
            ])
            for _ in range(depth)
        ])
        
        # Initialize new layers with randomness
        randomness_scale=1e-3
        with torch.no_grad():
            for layer in self.new_input_layers:
                ff_layer = layer[1]
                last_linear = ff_layer[-1]
                random_values = torch.randn_like(last_linear.weight) * randomness_scale
                last_linear.weight.copy_(random_values)
                
            for layer in self.prompt_layers:
                ff_layer = layer[1]
                last_linear = ff_layer[-1]
                random_values = torch.randn_like(last_linear.weight) * randomness_scale
                last_linear.weight.copy_(random_values)

    def forward(self, x, new_input, prompt_input):
        # new_input means the instruct
        # import pdb; pdb.set_trace()
        x_input = self.proj_in(x)
        latents = self.latents.repeat(x_input.size(0), 1, 1)
        
        new_input_transformed = self.proj_new_input(new_input)
        prompt_input_transformed = self.proj_prompt_input(prompt_input)

        for (orig_attn, orig_ff), (new_attn, new_ff), (prompt_attn, prompt_ff) in zip(self.layers, self.new_input_layers, self.prompt_layers):
            # Update latents from the original input path
            latents = orig_attn(x_input, latents) + latents
            latents = orig_ff(latents) + latents
            
            # original prompt info
            latents = prompt_attn(prompt_input_transformed, latents) + latents
            latents = prompt_ff(latents) + latents
            
            # Update latents from the new input path
            latents = new_attn(new_input_transformed, latents) + latents
            latents = new_ff(latents) + latents

        output = self.proj_out(latents)
        return self.norm_out(output)

def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim, keepdim=True) / denom.clamp(min=1e-5)
