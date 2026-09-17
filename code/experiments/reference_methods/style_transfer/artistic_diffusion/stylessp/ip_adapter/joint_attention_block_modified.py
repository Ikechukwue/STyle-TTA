# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class JointTransformerBlock_IP(nn.Module):
    """
    Transformer block used in SD3 (MMDiT).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        context_pre_only (`bool`):
            Boolean to determine if we should only apply the context processing.
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, context_pre_only: bool = False):
        super().__init__()

        self.context_pre_only = context_pre_only

        # 1. Self-Attn and Context-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=True,
            out_bias=True,
        )
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=True,
            out_bias=True,
        )

        # 2. Feed Forward
        if not context_pre_only:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff = FeedForward(dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim, dim_out=dim, activation_fn="gelu-approximate")

        # 3. AdaLN-single parameters
        self.scale_shift_table = nn.Parameter(torch.randn(6 * dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
    ):
        # 1. Input
        # hidden_states: [batch, length, dim]
        # encoder_hidden_states: [batch, length, dim]
        # temb: [batch, dim]

        # 2. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + temb.reshape(temb.shape[0], 6, -1)
        ).chunk(6, dim=1)

        # 3. Self-Attn and Context-Attn
        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_msa) + shift_msa

        # Attention
        attn_output, context_attn_output = self.attn(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )
        
        # IP-Adapter Attention
        attn_output_ip, context_attn_output_ip = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output + attn_output_ip

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output + context_attn_output_ip

        if self.context_pre_only:
            return encoder_hidden_states

        # 4. Feed Forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp * ff_output
        hidden_states = hidden_states + ff_output

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = gate_mlp * context_ff_output
        encoder_hidden_states = encoder_hidden_states + context_ff_output

        return encoder_hidden_states, hidden_states
