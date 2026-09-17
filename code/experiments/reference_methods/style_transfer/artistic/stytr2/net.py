"""
StyTr^2 (Image Style Transfer with Transformers) - INFERENCE ONLY

Paper: "StyTr^2: Image Style Transfer with Transformers"
Code:  https://github.com/diyiiyiii/StyTR-2

Architecture:
- VGG-19 encoder (up to relu5_1, 44 layers, frozen)
- PatchEmbed (8x8 patches, 512 embedding dim)
- Transformer with separate content & style encoders (3 layers each)
- Transformer decoder (3 layers)
- Content-Aware Positional Embedding (CAPE)
- Progressive upsampling decoder
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ─────────────────────────────────────────────────────────────────

def to_2tuple(x):
    return (x, x)


# ── VGG-19 Encoder ─────────────────────────────────────────────────────────

class VGGEncoder(nn.Module):
    """VGG-19 encoder (relu1_1 … relu5_1) with slice-based feature extraction."""

    def __init__(self):
        super(VGGEncoder, self).__init__()
        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),
        )
        self.slice2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
        )
        self.slice3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
        )
        self.slice4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),
        )
        self.slice5 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),
        )

    def forward(self, x, output_last_feature=False):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        if output_last_feature:
            return h5
        return [h1, h2, h3, h4, h5]


# ── Patch Embedding ─────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Image to Patch Embedding (8×8 patches, 512-d)."""

    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


# ── Transformer components ──────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        q = k = tgt if query_pos is None else tgt + query_pos
        v = tgt
        if memory is not None:
            k = memory if pos is None else memory + pos
            v = memory
            tgt2 = self.self_attn(q, k, value=v, attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        tgt2 = self.multihead_attn(query=q, key=k, value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    """Transformer with separate content/style encoders and CAPE."""

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder_c = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.encoder_s = TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers,
                                            copy.deepcopy(encoder_norm))
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.pos_embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((18, 18)),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
        )
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, content, style, content_pos=None, style_pos=None,
                mask=None, content_key_padding_mask=None, style_key_padding_mask=None):
        if content_pos is None:
            content_pos = F.interpolate(
                self.pos_embedding(content), size=content.shape[-2:], mode='bilinear'
            )
        if style_pos is None:
            style_pos = F.interpolate(
                self.pos_embedding(style), size=style.shape[-2:], mode='bilinear'
            )
        N, C, H, W = content.shape
        content = content.flatten(2).permute(2, 0, 1)
        style = style.flatten(2).permute(2, 0, 1)
        content_pos = content_pos.flatten(2).permute(2, 0, 1)
        style_pos = style_pos.flatten(2).permute(2, 0, 1)

        memory_c = self.encoder_c(content, mask=mask,
                                  src_key_padding_mask=content_key_padding_mask,
                                  pos=content_pos)
        memory_s = self.encoder_s(style, mask=mask,
                                  src_key_padding_mask=style_key_padding_mask,
                                  pos=style_pos)
        hs = self.decoder(memory_c, memory_s, memory_mask=mask,
                          memory_key_padding_mask=style_key_padding_mask,
                          pos=style_pos, query_pos=content_pos)
        return hs.permute(1, 2, 0).view(N, C, H, W)


# ── Progressive Decoder ────────────────────────────────────────────────────

class Decoder(nn.Module):
    """512-ch 32×32 → 3-ch 256×256 (3 stages of 2× upsampling)."""

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, x):
        return self.decoder(x)


# ── Main network ───────────────────────────────────────────────────────────

class StyTrans(nn.Module):
    """StyTr^2 — inference only."""

    def __init__(self):
        super(StyTrans, self).__init__()
        self.vgg = VGGEncoder()
        self.patch_embed = PatchEmbed(img_size=256, patch_size=8,
                                      in_chans=3, embed_dim=512)
        self.transformer = Transformer(d_model=512, nhead=8,
                                       num_encoder_layers=3,
                                       num_decoder_layers=3,
                                       dim_feedforward=2048, dropout=0.1)
        self.decoder = Decoder()

    def forward(self, content, style):
        content_tokens = self.patch_embed(content)
        style_tokens = self.patch_embed(style)
        stylized_feat = self.transformer(content_tokens, style_tokens)
        return self.decoder(stylized_feat)
