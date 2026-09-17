"""
xAILab Bamberg
University of Bamberg

@description:
AdaAttN (Adaptive Attentional Instance Normalization) network architecture.
Based on: https://github.com/Huage001/AdaAttN
"""

import torch
import torch.nn as nn


def calc_mean_std(feat, eps=1e-5):
    """Channel-wise mean and std for feature maps (N, C, H, W)."""
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """Normalise features to zero mean and unit variance."""
    mean, std = calc_mean_std(feat)
    return (feat - mean.expand(feat.size())) / std.expand(feat.size())


class AdaAttN(nn.Module):
    """Adaptive Attentional Normalization module."""

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super().__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)

        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()

        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].transpose(1, 2).contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()

        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)

        S = torch.bmm(F, G)
        S = self.sm(S)

        mean = torch.bmm(S, style_flat)
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))

        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):
    """Multi-scale attention transformer (relu4_1 + relu5_1)."""

    def __init__(self, in_planes, key_planes=None, shallow_layer=False):
        super().__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(
            in_planes=in_planes,
            key_planes=key_planes + 512 if shallow_layer else key_planes,
        )
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1,
                content4_1_key, style4_1_key, content5_1_key, style5_1_key,
                seed=None):
        return self.merge_conv(self.merge_conv_pad(
            self.attn_adain_4_1(content4_1, style4_1,
                                content4_1_key, style4_1_key, seed=seed) +
            self.upsample5_1(self.attn_adain_5_1(content5_1, style5_1,
                                                  content5_1_key, style5_1_key, seed=seed))
        ))


class Decoder(nn.Module):
    """Decoder mirroring VGG encoder: relu4_1 features -> RGB."""

    def __init__(self, skip_connection_3=False):
        super().__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        in_ch = 256 + 256 if skip_connection_3 else 256
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_ch, 256, (3, 3)),
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

    def forward(self, cs, c_adain_3_feat=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3_feat is not None:
            cs = self.decoder_layer_2(torch.cat((cs, c_adain_3_feat), dim=1))
        else:
            cs = self.decoder_layer_2(cs)
        return cs


# VGG-19 encoder (full, up to relu5-4) with normalised weights convention
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)


class AdaAttNNet(nn.Module):
    """Complete AdaAttN network: encoder + transformer + decoder."""

    def __init__(self, encoder, decoder, transformer, adaattn_3=None,
                 shallow_layer=True):
        super().__init__()
        self.encoder_layers = encoder   # plain list – not nn.ModuleList
        self.decoder = decoder
        self.transformer = transformer
        self.adaattn_3 = adaattn_3
        self.shallow_layer = shallow_layer

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            results.append(self.encoder_layers[i](results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(mean_variance_norm(
                    nn.functional.interpolate(feats[i], (h, w),
                                              mode='bilinear',
                                              align_corners=False)
                ))
            results.append(mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return mean_variance_norm(feats[last_layer_idx])

    def forward(self, content, style, seed=None):
        c_feats = self.encode_with_intermediate(content)
        s_feats = self.encode_with_intermediate(style)

        if self.adaattn_3 is not None:
            c_adain_feat_3 = self.adaattn_3(
                c_feats[2], s_feats[2],
                self.get_key(c_feats, 2, self.shallow_layer),
                self.get_key(s_feats, 2, self.shallow_layer),
                seed,
            )
        else:
            c_adain_feat_3 = None

        cs = self.transformer(
            c_feats[3], s_feats[3], c_feats[4], s_feats[4],
            self.get_key(c_feats, 3, self.shallow_layer),
            self.get_key(s_feats, 3, self.shallow_layer),
            self.get_key(c_feats, 4, self.shallow_layer),
            self.get_key(s_feats, 4, self.shallow_layer),
            seed,
        )

        return self.decoder(cs, c_adain_feat_3)
