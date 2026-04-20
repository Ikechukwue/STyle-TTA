"""
CAST Network Architecture (inference only)

xAILab Bamberg, University of Bamberg
Based on: https://github.com/zyxElsa/CAST_pytorch

Kept for inference:
- ADAIN_Encoder (VGG-based with AdaIN)
- Decoder (mirrored VGG decoder)

Removed (training only):
- StyleExtractor, Projector, InfoNCELoss (contrastive learning)
- PatchGANDiscriminator (GAN training)
- CASTNet composite wrapper
"""

import torch
import torch.nn as nn


# ============================================================================
# ADAIN Encoder (VGG up to relu4_1 with AdaIN)
# ============================================================================

class ADAIN_Encoder(nn.Module):
    """
    Encoder using Adaptive Instance Normalization.
    Uses pretrained VGG-19 (normalised) up to relu4_1.
    """

    def __init__(self, encoder):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])   # -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) # -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) # -> relu4_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, x):
        results = [x]
        for i in range(4):
            results.append(getattr(self, f'enc_{i+1}')(results[-1]))
        return results[1:]

    @staticmethod
    def _calc_mean_std(feat, eps=1e-5):
        N, C = feat.size()[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat, style_feat):
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        style_mean, style_std = self._calc_mean_std(style_feat)
        content_mean, content_std = self._calc_mean_std(content_feat)
        normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        return self.adain(content_feats[-1], style_feats[-1])


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """Mirror decoder: 512-ch feature map -> 3-ch RGB image."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, feat):
        return self.decoder(feat)
