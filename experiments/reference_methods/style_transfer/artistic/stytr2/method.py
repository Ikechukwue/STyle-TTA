"""
xAILab Bamberg
University of Bamberg

@description:
StyTr^2 method implementation - INFERENCE ONLY
Based on: https://github.com/diyiiyiii/StyTR-2

Paper: "StyTr^2: Image Style Transfer with Transformers"

Weight preparation:
Download from the official repository:
    - vgg_normalised.pth (VGG-19 encoder, 53 layers up to relu5-4)
    - decoder_iter_160000.pth (decoder weights)
    - transformer_iter_160000.pth (transformer weights)
    - embedding_iter_160000.pth (patch embed weights)
    From: https://github.com/diyiiyiii/StyTR-2

Merge into a single checkpoint:
python -c "
import torch, torch.nn as nn
import copy
import math
import torch.nn.functional as F


# ── helpers ─────────────────────────────────────────────────────────────────

def to_2tuple(x):
    return (x, x)


# ── VGG-19 Encoder ─────────────────────────────────────────────────────────

class VGGEncoder(nn.Module):
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

net = StyTrans()
# ── VGG encoder ──
# vgg_normalised.pth is a flat Sequential with 53 layers (up to relu5-4).
# Build matching flat model to load weights, then copy into slice-based VGGEncoder.
_flat_vgg = nn.Sequential(
    nn.Conv2d(3, 3, 1),
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(3, 64, 3), nn.ReLU(),      # relu1-1 (idx 3)
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(64, 64, 3), nn.ReLU(),     # relu1-2 (idx 6)
    nn.MaxPool2d((2,2),(2,2),(0,0), ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(64, 128, 3), nn.ReLU(),    # relu2-1 (idx 10)
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(128, 128, 3), nn.ReLU(),   # relu2-2 (idx 13)
    nn.MaxPool2d((2,2),(2,2),(0,0), ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(128, 256, 3), nn.ReLU(),   # relu3-1 (idx 17)
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 256, 3), nn.ReLU(),   # relu3-2
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 256, 3), nn.ReLU(),   # relu3-3
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 256, 3), nn.ReLU(),   # relu3-4
    nn.MaxPool2d((2,2),(2,2),(0,0), ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(256, 512, 3), nn.ReLU(),   # relu4-1 (idx 30)
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 512, 3), nn.ReLU(),   # relu4-2
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 512, 3), nn.ReLU(),   # relu4-3
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 512, 3), nn.ReLU(),   # relu4-4
    nn.MaxPool2d((2,2),(2,2),(0,0), ceil_mode=True),
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 512, 3), nn.ReLU(),   # relu5-1 (idx 43)
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 512, 3), nn.ReLU(),   # relu5-2
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 512, 3), nn.ReLU(),   # relu5-3
    nn.ReflectionPad2d((1,1,1,1)), nn.Conv2d(512, 512, 3), nn.ReLU(),   # relu5-4 (idx 52)
)
_flat_vgg.load_state_dict(torch.load('vgg_normalised.pth', weights_only=True))
flat_layers = list(_flat_vgg.children())
# Copy into slice-based VGGEncoder using official slice indices
s1 = nn.Sequential(*flat_layers[:4])
s2 = nn.Sequential(*flat_layers[4:11])
s3 = nn.Sequential(*flat_layers[11:18])
s4 = nn.Sequential(*flat_layers[18:31])
s5 = nn.Sequential(*flat_layers[31:44])
net.vgg.slice1.load_state_dict(s1.state_dict())
net.vgg.slice2.load_state_dict(s2.state_dict())
net.vgg.slice3.load_state_dict(s3.state_dict())
net.vgg.slice4.load_state_dict(s4.state_dict())
net.vgg.slice5.load_state_dict(s5.state_dict())
# ── Decoder ──
dec_sd = torch.load('decoder_iter_160000.pth', weights_only=True)
dec_sd = {'decoder.' + k: v for k, v in dec_sd.items()}
net.decoder.load_state_dict(dec_sd)
# ── Transformer ──
trans_sd = torch.load('transformer_iter_160000.pth', weights_only=True)
# Official checkpoint uses 'new_ps' for the positional embedding conv
trans_sd = {k.replace('new_ps.', 'pos_embedding.1.'): v for k, v in trans_sd.items()}
# encoder_c/s norms were not saved in the official checkpoint; strict=False keeps LayerNorm defaults
net.transformer.load_state_dict(trans_sd, strict=False)
# ── Patch Embedding ──
embed_sd = torch.load('embedding_iter_160000.pth', weights_only=True)
net.patch_embed.load_state_dict(embed_sd)
# ── Save merged ──
torch.save(net.state_dict(), 'stytr2.pth')
print('Saved merged weights to stytr2.pth')
"
"""

from pathlib import Path
import torch

from experiments.reference_methods.style_transfer.artistic.stytr2.net import StyTrans


class Method:
    """
    StyTr^2 style transfer method - INFERENCE ONLY.

    Paper: "StyTr^2: Image Style Transfer with Transformers"
    """

    def __init__(
        self,
        pretrained_weights: str = None,
        device: str = 'cpu',
    ):
        self.network = None
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError(
                "pretrained_weights must point to a merged .pth file "
                "(see module docstring for merge script)"
            )

        self._initialize_network(Path(pretrained_weights))

    # ── configuration ───────────────────────────────────────────────────────
    def get_default_config(self) -> dict:
        """
        Default configuration from official StyTr^2 train.py.
        Kept for reference / reproducibility.
        """
        return {
            'learning_rate': 5e-4,
            'lr_decay': 1e-5,
            'batch_size': 8,
            'max_iter': 160000,
            'style_weight': 10.0,
            'content_weight': 7.0,
            'save_interval': 10000,
        }

    @staticmethod
    def get_native_image_size() -> int:
        """Native resolution (256×256)."""
        return 256

    # ── inference ───────────────────────────────────────────────────────────
    def __call__(self, content, style):
        """
        Perform style transfer.

        Args:
            content: (B, 3, H, W) tensor in [0, 1]
            style:   (B, 3, H, W) tensor in [0, 1]

        Returns:
            Stylized image (B, 3, H, W) clamped to [0, 1]
        """
        if self.network is None:
            raise RuntimeError("Network not initialized.")

        with torch.no_grad():
            output = self.network(content, style)

        return output.clamp(0, 1)

    # ── weight loading ──────────────────────────────────────────────────────
    def _initialize_network(self, weights_path: Path):
        """
        Build StyTrans and load the full merged state_dict.

        Expected: a single .pth file produced by the merge script
        (contains vgg + patch_embed + transformer + decoder keys).
        """
        self.network = StyTrans()

        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        # Handle wrapped checkpoints
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.network.load_state_dict(state_dict)
        self.network.to(self.device)

        for p in self.network.parameters():
            p.requires_grad = False

        self.network.eval()
