"""
IEContrAST Network Architecture (inference only)

xAILab Bamberg, University of Bamberg
Based on: https://github.com/HalbertCH/IEContraAST

Kept for inference:
- SANet (style-attentional module)
- Transform (multi-level SANet at relu4_1 + relu5_1)
- IEContrASTNet (VGG encoder + Transform + decoder)

Removed (training only):
- projection heads, contrastive losses, MultiDiscriminator, identity losses
"""

import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    return (feat - mean.expand(size)) / std.expand(size)


class SANet(nn.Module):
    """Style-Attentional Network: attention-based content-style alignment."""

    def __init__(self, in_planes):
        super().__init__()
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
        S = self.sm(torch.bmm(F, G))
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        return O + content


class Transform(nn.Module):
    """Multi-level SANet transformation at relu4_1 and relu5_1."""

    def __init__(self, in_planes):
        super().__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        upsample = nn.Upsample(
            size=(content4_1.size(2), content4_1.size(3)), mode='nearest'
        )
        return self.merge_conv(self.merge_conv_pad(
            self.sanet4_1(content4_1, style4_1) +
            upsample(self.sanet5_1(content5_1, style5_1))
        ))


class IEContrASTNet(nn.Module):
    """
    IEContrAST: VGG encoder (frozen) + SANet Transform + decoder.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])    # -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])   # -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # -> relu5_1
        self.transform = Transform(in_planes=512)
        self.decoder = decoder

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, x):
        results = [x]
        for i in range(5):
            results.append(getattr(self, f'enc_{i+1}')(results[-1]))
        return results[1:]
