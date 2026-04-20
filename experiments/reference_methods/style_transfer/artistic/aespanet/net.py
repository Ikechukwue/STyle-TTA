"""
xAILab Bamberg
University of Bamberg

@description:
AesPA-Net (Aesthetic Pattern-Aware Style Transfer Networks) network architecture.
Based on: https://github.com/Kibeom-Hong/AesPA-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    mean, std = calc_mean_std(feat)
    return (feat - mean.expand(feat.size())) / std.expand(feat.size())


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (ch * h * w)


def contextual_loss_v2(x, y, h=0.5):
    N, C, H, W = x.size()
    x_flat = F.normalize(x.view(N, C, -1).permute(0, 2, 1), p=2, dim=2)
    y_flat = F.normalize(y.view(N, C, -1).permute(0, 2, 1), p=2, dim=2)
    d = 1 - torch.bmm(x_flat, y_flat.transpose(1, 2))
    d_min, _ = torch.min(d, dim=2, keepdim=True)
    d_tilde = d / (d_min + 1e-5)
    w = torch.exp((1 - d_tilde) / h)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)
    return torch.mean(-torch.log(cx + 1e-5))


# ---- Attention ----

class AdaptiveMultiAdaAttN_v2(nn.Module):
    def __init__(self, in_planes, out_planes, max_sample=256*256,
                 query_planes=None, key_planes=None):
        super().__init__()
        if query_planes is None:
            query_planes = in_planes
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(query_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, out_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample

    def forward(self, content, style, content_key, style_key, seed=None):
        F_q = self.f(content_key)
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
        b, _, h, w = F_q.size()
        F_q = F_q.view(b, -1, w * h).permute(0, 2, 1)
        S = self.sm(torch.bmm(F_q, G))
        mean = torch.bmm(S, style_flat)
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat ** 2) - mean ** 2))
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean, S


class AdaptiveMultiAttn_Transformer_v2(nn.Module):
    def __init__(self, in_planes, out_planes, query_planes=None,
                 key_planes=None, shallow_layer=False):
        super().__init__()
        self.attn_adain_4_1 = AdaptiveMultiAdaAttN_v2(
            in_planes, out_planes, query_planes=query_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaptiveMultiAdaAttN_v2(
            in_planes, out_planes, query_planes=query_planes,
            key_planes=key_planes + 512)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, c4, s4, c5, s5, ck4, sk4, ck5, sk5, seed=None):
        f4, a4 = self.attn_adain_4_1(c4, s4, ck4, sk4, seed=seed)
        f5, a5 = self.attn_adain_5_1(c5, s5, ck5, sk5, seed=seed)
        merged = self.merge_conv(self.merge_conv_pad(
            f4 + F.interpolate(f5, size=(f4.size(2), f4.size(3)), mode='nearest')
        ))
        return merged, f4, f5, a4, a5


# ---- AdaIN helper ----

def feature_wct_simple(content_feat, style_feat, alpha=1.0):
    size = content_feat.size()
    s_mean, s_std = calc_mean_std(style_feat)
    c_mean, c_std = calc_mean_std(content_feat)
    norm = (content_feat - c_mean.expand(size)) / c_std.expand(size)
    stylized = norm * s_std.expand(size) + s_mean.expand(size)
    return alpha * stylized + (1 - alpha) * content_feat


# ---- VGG Encoder ----

class VGGEncoder(nn.Module):
    """VGG-19 encoder loading VGG normalised weights (pth format)."""

    def __init__(self, state_dict=None):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2)

        # Declare all conv layers
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv4_4 = nn.Conv2d(512, 512, 3, 1, 0)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 0)

        if state_dict is not None:
            self._load_from_sequential_dict(state_dict)

    def _load_from_sequential_dict(self, sd):
        """Load from nn.Sequential-style numeric keys (e.g. '0.weight')."""
        mapping = {
            'conv0': '0', 'conv1_1': '2', 'conv1_2': '5',
            'conv2_1': '9', 'conv2_2': '12',
            'conv3_1': '16', 'conv3_2': '19', 'conv3_3': '22', 'conv3_4': '25',
            'conv4_1': '29', 'conv4_2': '32', 'conv4_3': '35', 'conv4_4': '38',
            'conv5_1': '42',
        }
        for attr, idx in mapping.items():
            getattr(self, attr).weight = nn.Parameter(sd[f'{idx}.weight'].clone())
            getattr(self, attr).bias = nn.Parameter(sd[f'{idx}.bias'].clone())

    def encode(self, x, skips):
        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)));  skips['conv1_1'] = out
        out = self.relu(self.conv1_2(self.pad(out)));  skips['conv1_2'] = out
        rw, rh = out.size(2), out.size(3)
        p = self.pool(out);  skips['pool1'] = out - F.interpolate(p, size=[rw, rh], mode='nearest')

        out = self.relu(self.conv2_1(self.pad(p)));    skips['conv2_1'] = out
        out = self.relu(self.conv2_2(self.pad(out)));  skips['conv2_2'] = out
        rw, rh = out.size(2), out.size(3)
        p = self.pool(out);  skips['pool2'] = out - F.interpolate(p, size=[rw, rh], mode='nearest')

        out = self.relu(self.conv3_1(self.pad(p)));    skips['conv3_1'] = out
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)));  skips['conv3_4'] = out
        rw, rh = out.size(2), out.size(3)
        p = self.pool(out);  skips['pool3'] = out - F.interpolate(p, size=[rw, rh], mode='nearest')

        out = self.relu(self.conv4_1(self.pad(p)));    skips['conv4_1'] = out
        out = self.relu(self.conv4_2(self.pad(out)))
        out = self.relu(self.conv4_3(self.pad(out)))
        out = self.relu(self.conv4_4(self.pad(out)));  skips['conv4_4'] = out
        rw, rh = out.size(2), out.size(3)
        p = self.pool(out);  skips['pool4'] = out - F.interpolate(p, size=[rw, rh], mode='nearest')

        out = self.relu(self.conv5_1(self.pad(p)));    skips['conv5_1'] = out
        return out

    def get_features(self, x, level):
        out = self.conv0(x)
        out = self.relu(self.conv1_1(self.pad(out)))
        if level == 1:
            return out
        out = self.relu(self.conv1_2(self.pad(out)))
        out = self.relu(self.conv2_1(self.pad(self.pool(out))))
        if level == 2:
            return out
        out = self.relu(self.conv2_2(self.pad(out)))
        out = self.relu(self.conv3_1(self.pad(self.pool(out))))
        if level == 3:
            return out
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_4(self.pad(out)))
        out = self.relu(self.conv4_1(self.pad(self.pool(out))))
        if level == 4:
            return out
        out = self.relu(self.conv4_2(self.pad(out)))
        out = self.relu(self.conv4_3(self.pad(out)))
        out = self.relu(self.conv4_4(self.pad(out)))
        out = self.relu(self.conv5_1(self.pad(self.pool(out))))
        return out


# ---- VGG Decoder ----

class VGGDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=False)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def decode(self, feat, content_skips, style_skips):
        out = self.relu(self.conv4_1(self.pad(feat)))
        sz = (content_skips['conv3_4'].size(2), content_skips['conv3_4'].size(3))
        out = F.interpolate(out, size=sz, mode='nearest')
        out = self.relu(self.conv3_4(self.pad(out)))
        out = self.relu(self.conv3_3(self.pad(out)))
        out = self.relu(self.conv3_2(self.pad(out)))
        out = self.relu(self.conv3_1(self.pad(out)))
        sz = (content_skips['conv2_2'].size(2), content_skips['conv2_2'].size(3))
        out = F.interpolate(out, size=sz, mode='nearest')
        out = self.relu(self.conv2_2(self.pad(out)))
        out = self.relu(self.conv2_1(self.pad(out)))
        sz = (content_skips['conv1_2'].size(2), content_skips['conv1_2'].size(3))
        out = F.interpolate(out, size=sz, mode='nearest')
        out = self.relu(self.conv1_2(self.pad(out)))
        return self.conv1_1(self.pad(out))


# ---- Complete AesPA-Net ----

class AesPANet(nn.Module):
    def __init__(self, encoder_state_dict=None):
        super().__init__()
        self.encoder = VGGEncoder(state_dict=encoder_state_dict)
        self.decoder = VGGDecoder()
        query_ch = 512
        key_ch = 512 + 256 + 128 + 64  # 960
        self.transformer = AdaptiveMultiAttn_Transformer_v2(
            in_planes=512, out_planes=512,
            query_planes=query_ch, key_planes=key_ch)

    def adaptive_get_keys(self, skips, start, end, target_feat):
        _, _, th, tw = target_feat.shape
        target_key = f'conv{end}_1'
        _, _, h, w = skips[target_key].shape
        parts = []
        for i in range(start, end + 1):
            f = skips[f'conv{i}_1']
            if i == end:
                parts.append(mean_variance_norm(f))
            else:
                parts.append(mean_variance_norm(F.interpolate(f, (h, w), mode='nearest')))
        return F.interpolate(torch.cat(parts, dim=1), (th, tw), mode='nearest')

    def forward(self, content, style, adaptive_alpha, gray_content=None, gray_style=None):
        c_sk, s_sk = {}, {}
        self.encoder.encode(content, c_sk)
        self.encoder.encode(style, s_sk)

        key_c_sk = c_sk
        key_s_sk = s_sk

        local_feat, f4, f5, a4, a5 = self.transformer(
            c_sk['conv4_1'], s_sk['conv4_1'],
            c_sk['conv5_1'], s_sk['conv5_1'],
            self.adaptive_get_keys(key_c_sk, 4, 4, c_sk['conv4_1']),
            self.adaptive_get_keys(key_s_sk, 1, 4, s_sk['conv4_1']),
            self.adaptive_get_keys(key_c_sk, 5, 5, c_sk['conv5_1']),
            self.adaptive_get_keys(key_s_sk, 1, 5, s_sk['conv5_1']),
        )

        if gray_content is not None:
            gc_sk = {}
            self.encoder.encode(gray_content, gc_sk)
            if gray_style is not None:
                gs_sk = {}
                self.encoder.encode(gray_style, gs_sk)
                global_feat = feature_wct_simple(gc_sk['conv4_1'], gs_sk['conv4_1'])
            else:
                global_feat = feature_wct_simple(gc_sk['conv4_1'], s_sk['conv4_1'])
        else:
            global_feat = feature_wct_simple(c_sk['conv4_1'], s_sk['conv4_1'])

        if len(adaptive_alpha.shape) == 2:
            adaptive_alpha = adaptive_alpha.unsqueeze(-1).unsqueeze(-1)
        adaptive_alpha = adaptive_alpha.to(global_feat.device)

        blended = global_feat * (1 - adaptive_alpha) + adaptive_alpha * local_feat
        out = self.decoder.decode(blended, c_sk, s_sk)
        return out, f4, f5, a4, a5
