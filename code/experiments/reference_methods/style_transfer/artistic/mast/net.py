"""
MAST Network Architecture (inference only)

xAILab Bamberg, University of Bamberg
Based on: https://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network

Kept: CA, Content_SA, Style_SA, Multi_Adaptation_Module, MASTNet
Removed: compute_loss, shuffle, calc_content_loss, calc_style_loss
"""

import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normal(feat, eps=1e-5):
    mean, std = calc_mean_std(feat, eps)
    return (feat - mean) / std


class CA(nn.Module):
    """Cross-Attention between content and style features."""
    def __init__(self, in_dim):
        super().__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()
        F_norm = self.f(normal(content_feat)).view(B, -1, H*W).permute(0, 2, 1)
        G_norm = self.g(normal(style_feat)).view(B, -1, H*W)
        attention = self.softmax(torch.bmm(F_norm, G_norm))
        H_s = self.h(style_feat).view(B, -1, H*W)
        out = torch.bmm(H_s, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.out_conv(out) + content_feat


class Style_SA(nn.Module):
    """Channel-wise style self-attention."""
    def __init__(self, in_dim):
        super().__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, style_feat):
        B, C, H, W = style_feat.size()
        F_s = self.f(style_feat).view(B, -1, H*W)
        G_s = self.g(style_feat).view(B, -1, H*W).permute(0, 2, 1)
        attention = self.softmax(torch.bmm(F_s, G_s))
        H_s = self.h(normal(style_feat)).view(B, -1, H*W)
        out = torch.bmm(attention.permute(0, 2, 1), H_s).view(B, C, H, W)
        return self.out_conv(out) + style_feat


class Content_SA(nn.Module):
    """Position-wise content self-attention."""
    def __init__(self, in_dim):
        super().__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat):
        B, C, H, W = content_feat.size()
        F_norm = self.f(normal(content_feat)).view(B, -1, H*W).permute(0, 2, 1)
        G_norm = self.g(normal(content_feat)).view(B, -1, H*W)
        attention = self.softmax(torch.bmm(F_norm, G_norm))
        H_c = self.h(content_feat).view(B, -1, H*W)
        out = torch.bmm(H_c, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.out_conv(out) + content_feat


class Multi_Adaptation_Module(nn.Module):
    """Content SA + Style SA + Cross-Attention at relu4_1."""
    def __init__(self, in_dim):
        super().__init__()
        self.CA = CA(in_dim)
        self.CSA = Content_SA(in_dim)
        self.SSA = Style_SA(in_dim)

    def forward(self, content_feats, style_feats):
        content_feat = self.CSA(content_feats[-2])
        style_feat = self.SSA(style_feats[-2])
        return self.CA(content_feat, style_feat)


class MASTNet(nn.Module):
    """MAST: VGG encoder + Multi-Adaptation Module + decoder."""
    def __init__(self, encoder, decoder):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.enc_5 = nn.Sequential(*enc_layers[31:44])
        self.ma_module = Multi_Adaptation_Module(512)
        self.decoder = decoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, x):
        results = [x]
        for i in range(5):
            results.append(getattr(self, f'enc_{i+1}')(results[-1]))
        return results[1:]

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        feat = self.ma_module(content_feats, style_feats)
        return self.decoder(feat)
