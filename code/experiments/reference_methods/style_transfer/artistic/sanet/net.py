"""
xAILab Bamberg
University of Bamberg

@description:
SANET (Style-Attentional Network) network architecture - INFERENCE ONLY
Based on: https://github.com/GlebSBrykin/SANET

Paper: "Arbitrary Style Transfer with Style-Attentional Networks"
Authors: Dae Young Park, Kwang Hee Lee
Conference: CVPR 2019
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
    Style-Attentional Network module.

    Computes attention-weighted style features using three 1x1 convolutions
    (f, g, h) for query, key, value, followed by softmax attention.
    """

    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
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
        O = torch.bmm(H, S.permute(0, 2, 1))

        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O


class Transform(nn.Module):
    """
    Transformation module combining two SANet modules at relu4_1 and relu5_1.
    Merges results with upsampling and a merge convolution.
    """

    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        out4_1 = self.sanet4_1(content4_1, style4_1)
        out5_1 = self.sanet5_1(content5_1, style5_1)
        out5_1_upsampled = self.upsample5_1(out5_1)
        merged = out4_1 + out5_1_upsampled
        output = self.merge_conv(self.merge_conv_pad(merged))
        return output


class SANETNet(nn.Module):
    """
    Complete SANET network for inference.

    Combines VGG encoder (frozen), SANet transformer, and decoder.
    """

    def __init__(self, encoder, decoder):
        super(SANETNet, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])    # -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])   # -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])   # -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])   # -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])   # -> relu5_1

        self.transform = Transform(in_planes=512)
        self.decoder = decoder

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, content, style):
        content_feats = self.encode_with_intermediate(content)
        style_feats = self.encode_with_intermediate(style)

        stylized = self.transform(
            content_feats[3], style_feats[3],
            content_feats[4], style_feats[4]
        )

        output = self.decoder(stylized)
        return output
