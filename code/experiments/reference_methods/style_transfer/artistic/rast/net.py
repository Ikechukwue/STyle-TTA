"""
xAILab Bamberg
University of Bamberg

@description:
RAST (Restorable Arbitrary Style Transfer) network architecture - INFERENCE ONLY
Based on: https://github.com/YingnanMa/RAST

Paper: "RAST: Restorable Arbitrary Style Transfer via Multi-restoration"
Authors: Yingnan Ma, Xudong Li et al.
Conference: WACV 2023

Architecture overview:
  - VGG-19 encoder (normalised, up to relu5_1, 44 layers)
  - SANet-based Transform module (attention at relu4_1 and relu5_1, merged)
  - Symmetric decoder (mirrors VGG from relu4_1 back to RGB)

The inference forward pass is:
  1. Encode content -> content_feat4_1, content_feat5_1
  2. Encode style   -> style_feat4_1, style_feat5_1
  3. Transform(c4_1, s4_1, c5_1, s5_1) -> stylized features
  4. Decode -> output image

Note: The network structure (SANet attention + VGG encoder/decoder) is shared with
SANet (CVPR 2019). The key difference is the training strategy (multi-restoration,
style-difference loss, contrastive learning). Inference code is architecturally
identical; only the learned weights differ.
"""

import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    """Calculate channel-wise mean and std for feature maps."""
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """Normalize features to zero mean and unit variance."""
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class SANet(nn.Module):
    """
    Style-Attentional Network module (from SANet / RAST).

    Computes attention between content queries and style keys/values
    using three 1x1 convolutions (f, g, h) followed by softmax attention.
    Content and style features are first normalised to zero mean / unit variance
    for query (f) and key (g); value (h) operates on un-normalised style features.
    A residual connection adds the original content features.
    """

    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))       # query (content)
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))       # key   (style)
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))       # value (style)
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)

        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)

        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)

        S = torch.bmm(F, G)
        S = self.sm(S)

        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        out = torch.bmm(H, S.permute(0, 2, 1))

        b, c, h, w = content.size()
        out = out.view(b, c, h, w)
        out = self.out_conv(out)
        out += content  # residual
        return out


class Transform(nn.Module):
    """
    Transformation module combining two SANet modules at relu4_1 and relu5_1.
    Merges results (with dynamic upsampling of relu5_1 features to relu4_1 size)
    through a 3x3 convolution.

    Note: The original RAST code re-creates nn.Upsample inside forward() to
    dynamically match spatial dimensions. We replicate this behaviour so the
    module works with arbitrary input resolutions.
    """

    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        out4_1 = self.sanet4_1(content4_1, style4_1)
        out5_1 = self.sanet5_1(content5_1, style5_1)
        # Dynamic upsample to match relu4_1 spatial size (as in original code)
        upsample = nn.Upsample(
            size=(content4_1.size(2), content4_1.size(3)),
            mode='nearest',
        )
        out5_1_up = upsample(out5_1)
        merged = out4_1 + out5_1_up
        return self.merge_conv(self.merge_conv_pad(merged))


class RASTNet(nn.Module):
    """
    Complete RAST network for inference.

    Combines:
      - VGG-19 encoder (frozen, split into enc_1..enc_5)
      - SANet-based Transform module
      - Symmetric decoder
    """

    def __init__(self, encoder, decoder):
        super(RASTNet, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])     # input   -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])    # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])   # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])   # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])   # relu4_1 -> relu5_1

        self.transform = Transform(in_planes=512)
        self.decoder = decoder

    def encode_with_intermediate(self, x):
        """Extract relu1_1 … relu5_1 feature maps."""
        results = [x]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, x):
        """Encode to relu4_1 and relu5_1 only."""
        feat = x
        for i in range(1, 5):
            feat = getattr(self, 'enc_{:d}'.format(i))(feat)
        feat4_1 = feat
        feat5_1 = self.enc_5(feat4_1)
        return feat4_1, feat5_1

    def forward(self, content, style):
        """
        Single-step style transfer.

        Args:
            content: (B, 3, H, W) in [0, 1]
            style:   (B, 3, H, W) in [0, 1]

        Returns:
            Stylized image (B, 3, H, W)
        """
        c4_1, c5_1 = self.encode(content)
        s4_1, s5_1 = self.encode(style)
        stylized = self.transform(c4_1, s4_1, c5_1, s5_1)
        return self.decoder(stylized)
