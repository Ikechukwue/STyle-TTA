"""
AesFA – Aesthetic Feature-Aware Arbitrary Neural Style Transfer (AAAI 2024).

PyTorch implementation from the official code:
    https://github.com/Sooyyoungg/AesFA

The network uses Octave Convolutions for frequency-based feature decomposition
(high/low frequency paths), with adaptive convolutions (KernelPredictor +
AdaConv2d) in the decoder for style injection.  Fully end-to-end; no pre-trained
VGG is required at inference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Octave Convolution primitives
# ══════════════════════════════════════════════════════════════════════════════

class Oct_conv_lreLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        return super().forward(hf), super().forward(lf)


class Oct_conv_up(nn.Upsample):
    def forward(self, x):
        hf, lf = x
        return super().forward(hf), super().forward(lf)


class Oct_Conv_aftup(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride, padding,
                 pad_type='reflect', alpha_in=0.5, alpha_out=0.5):
        super().__init__()
        lf_in = int(in_ch * alpha_in)
        lf_out = int(out_ch * alpha_out)
        hf_in = in_ch - lf_in
        hf_out = out_ch - lf_out
        self.conv_h = nn.Conv2d(hf_in, hf_out, ks, stride, padding,
                                padding_mode=pad_type)
        self.conv_l = nn.Conv2d(lf_in, lf_out, ks, stride, padding,
                                padding_mode=pad_type)

    def forward(self, x):
        hf, lf = x
        return self.conv_h(hf), self.conv_l(lf)


class OctConv(nn.Module):
    """Octave Convolution with high/low frequency paths."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, pad_type='reflect',
                 alpha_in=0.5, alpha_out=0.5,
                 type='normal', freq_ratio=(1, 1)):
        super().__init__()
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_in = int(in_channels * (1 - alpha_in))
        hf_out = int(out_channels * (1 - alpha_out))
        lf_in = in_channels - hf_in
        lf_out = out_channels - hf_out

        self.avg_pool = nn.AvgPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        is_dw = (groups == in_channels)

        if type == 'first':
            self.convh = nn.Conv2d(in_channels, hf_out, kernel_size,
                                   stride, padding, padding_mode=pad_type, bias=False)
            self.convl = nn.Conv2d(in_channels, lf_out, kernel_size,
                                   stride, padding, padding_mode=pad_type, bias=False)
        elif type == 'last':
            self.convh = nn.Conv2d(hf_in, out_channels, kernel_size,
                                   stride, padding, padding_mode=pad_type, bias=False)
            self.convl = nn.Conv2d(lf_in, out_channels, kernel_size,
                                   stride, padding, padding_mode=pad_type, bias=False)
        else:
            self.L2L = nn.Conv2d(lf_in, lf_out, kernel_size, stride, padding,
                                 groups=math.ceil(alpha_in * groups),
                                 padding_mode=pad_type, bias=False)
            self.H2H = nn.Conv2d(hf_in, hf_out, kernel_size, stride, padding,
                                 groups=math.ceil(groups - alpha_in * groups),
                                 padding_mode=pad_type, bias=False)
            if is_dw:
                self.L2H = None
                self.H2L = None
            else:
                self.L2H = nn.Conv2d(lf_in, hf_out, kernel_size, stride, padding,
                                     groups=groups, padding_mode=pad_type, bias=False)
                self.H2L = nn.Conv2d(hf_in, lf_out, kernel_size, stride, padding,
                                     groups=groups, padding_mode=pad_type, bias=False)

    def forward(self, x):
        if self.type == 'first':
            hf = self.convh(x)
            lf = self.convl(self.avg_pool(x))
            return hf, lf
        elif self.type == 'last':
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            output = out_h * self.freq_ratio[0] + out_l * self.freq_ratio[1]
            return output, out_h, out_l
        else:
            hf, lf = x
            if self.L2H is None:  # depthwise
                return self.H2H(hf), self.L2L(lf)
            return (self.H2H(hf) + self.L2H(self.upsample(lf)),
                    self.L2L(lf) + self.H2L(self.avg_pool(hf)))


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptive Octave Convolution (decoder building block)
# ══════════════════════════════════════════════════════════════════════════════

class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups,
                 style_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size
        pad = math.ceil((kernel_size - 1) / 2)
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size, padding=pad, padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(style_channels, out_channels * out_channels // n_groups, 1))
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(style_channels, out_channels, 1))

    def forward(self, w):
        w_sp = self.spatial(w).reshape(
            len(w), self.out_channels, self.in_channels // self.n_groups,
            self.kernel_size, self.kernel_size)
        w_pw = self.pointwise(w).reshape(
            len(w), self.out_channels, self.out_channels // self.n_groups, 1, 1)
        b = self.bias(w).reshape(len(w), self.out_channels)
        return w_sp, w_pw, b


class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super().__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        pad = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=(math.ceil(pad), math.floor(pad)),
                              padding_mode='reflect')

    def forward(self, x, w_sp, w_pw, bias):
        x = F.instance_norm(x)
        ys = []
        for i in range(len(x)):
            y = self._single(x[i:i + 1], w_sp[i], w_pw[i], bias[i])
            ys.append(y)
        return self.conv(torch.cat(ys, 0))

    def _single(self, x, w_sp, w_pw, bias):
        pad_val = (w_sp.size(-1) - 1) / 2
        pad = (math.ceil(pad_val), math.floor(pad_val),
               math.ceil(pad_val), math.floor(pad_val))
        x = F.pad(x, pad, mode='reflect')
        x = F.conv2d(x, w_sp, groups=self.n_groups)
        x = F.conv2d(x, w_pw, groups=self.n_groups, bias=bias)
        return x


class AdaOctConv(nn.Module):
    """Adaptive Octave Convolution: predicts kernels from style, applies to content."""

    def __init__(self, in_channels, out_channels, group_div,
                 style_channels, kernel_size, stride, padding,
                 oct_groups, alpha_in, alpha_out, type='normal'):
        super().__init__()
        h_in = int(in_channels * (1 - alpha_in))
        l_in = in_channels - h_in
        n_g_h = h_in // group_div
        n_g_l = l_in // group_div
        s_ch_h = int(style_channels * (1 - alpha_in))
        s_ch_l = style_channels - s_ch_h
        ks_h, ks_l, ks_A = kernel_size

        self.kernelPredictor_h = KernelPredictor(h_in, h_in, n_g_h, s_ch_h, ks_h)
        self.kernelPredictor_l = KernelPredictor(l_in, l_in, n_g_l, s_ch_l, ks_l)
        self.AdaConv_h = AdaConv2d(h_in, h_in, n_groups=n_g_h)
        self.AdaConv_l = AdaConv2d(l_in, l_in, n_groups=n_g_l)
        self.OctConv = OctConv(in_channels, out_channels, ks_A,
                               stride, padding, oct_groups,
                               alpha_in=alpha_in, alpha_out=alpha_out, type=type)
        self.relu = Oct_conv_lreLU()

    def forward(self, content, style, cond='train'):
        c_hf, c_lf = content
        s_hf, s_lf = style
        h_ws, h_wp, h_b = self.kernelPredictor_h(s_hf)
        l_ws, l_wp, l_b = self.kernelPredictor_l(s_lf)
        out_h = self.AdaConv_h(c_hf, h_ws, h_wp, h_b)
        out_l = self.AdaConv_l(c_lf, l_ws, l_wp, l_b)
        out = self.relu((out_h, out_l))
        out = self.OctConv(out)
        if isinstance(out, tuple) and len(out) == 2:
            out = self.relu(out)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder & Decoder
# ══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, in_dim=3, nf=64, style_kernel=(3, 3),
                 alpha_in=0.5, alpha_out=0.5):
        super().__init__()
        sk = style_kernel
        self.conv = nn.Conv2d(in_dim, nf, 7, 1, 3)
        self.OctConv1_1 = OctConv(nf, nf, 3, 2, 1, groups=nf,
                                  alpha_in=alpha_in, alpha_out=alpha_out, type='first')
        self.OctConv1_2 = OctConv(nf, 2 * nf, 1, alpha_in=alpha_in, alpha_out=alpha_out)
        self.OctConv1_3 = OctConv(2 * nf, 2 * nf, 3, 1, 1,
                                  alpha_in=alpha_in, alpha_out=alpha_out)
        self.OctConv2_1 = OctConv(2 * nf, 2 * nf, 3, 2, 1, groups=2 * nf,
                                  alpha_in=alpha_in, alpha_out=alpha_out)
        self.OctConv2_2 = OctConv(2 * nf, 4 * nf, 1, alpha_in=alpha_in, alpha_out=alpha_out)
        self.OctConv2_3 = OctConv(4 * nf, 4 * nf, 3, 1, 1,
                                  alpha_in=alpha_in, alpha_out=alpha_out)
        self.pool_h = nn.AdaptiveAvgPool2d(sk[0])
        self.pool_l = nn.AdaptiveAvgPool2d(sk[1])
        self.relu = Oct_conv_lreLU()

    def forward_content(self, x):
        out = self.conv(x)
        out = self.relu(self.OctConv1_1(out))
        out = self.relu(self.OctConv1_2(out))
        out = self.relu(self.OctConv1_3(out))
        out = self.relu(self.OctConv2_1(out))
        out = self.relu(self.OctConv2_2(out))
        out = self.relu(self.OctConv2_3(out))
        return out

    def forward_style(self, x):
        out = self.conv(x)
        out = self.relu(self.OctConv1_1(out))
        out = self.relu(self.OctConv1_2(out))
        out = self.relu(self.OctConv1_3(out))
        out = self.relu(self.OctConv2_1(out))
        out = self.relu(self.OctConv2_2(out))
        out = self.relu(self.OctConv2_3(out))
        hf, lf = out
        return self.pool_h(hf), self.pool_l(lf)

    def forward_test(self, x, cond):
        """Unified test forward matching original code."""
        out = self.conv(x)
        out = self.relu(self.OctConv1_1(out))
        out = self.relu(self.OctConv1_2(out))
        out = self.relu(self.OctConv1_3(out))
        out = self.relu(self.OctConv2_1(out))
        out = self.relu(self.OctConv2_2(out))
        out = self.relu(self.OctConv2_3(out))
        if cond == 'style':
            hf, lf = out
            return self.pool_h(hf), self.pool_l(lf)
        return out


class Decoder(nn.Module):
    def __init__(self, nf=64, out_dim=3, style_channel=256,
                 style_kernel=(3, 3, 3), alpha_in=0.5, alpha_out=0.5,
                 freq_ratio=(1, 1), pad_type='reflect'):
        super().__init__()
        group_div = [1, 2, 4, 8]
        sk = list(style_kernel)
        self.up_oct = Oct_conv_up(scale_factor=2)

        self.AdaOctConv1_1 = AdaOctConv(
            4 * nf, 4 * nf, group_div[0], style_channel, sk, 1, 1,
            4 * nf, alpha_in, alpha_out)
        self.OctConv1_2 = OctConv(4 * nf, 2 * nf, 1, 1,
                                  alpha_in=alpha_in, alpha_out=alpha_out)
        self.oct_conv_aftup_1 = Oct_Conv_aftup(2 * nf, 2 * nf, 3, 1, 1,
                                               pad_type, alpha_in, alpha_out)

        self.AdaOctConv2_1 = AdaOctConv(
            2 * nf, 2 * nf, group_div[1], style_channel, sk, 1, 1,
            2 * nf, alpha_in, alpha_out)
        self.OctConv2_2 = OctConv(2 * nf, nf, 1, 1,
                                  alpha_in=alpha_in, alpha_out=alpha_out)
        self.oct_conv_aftup_2 = Oct_Conv_aftup(nf, nf, 3, 1, 1,
                                               pad_type, alpha_in, alpha_out)

        self.AdaOctConv3_1 = AdaOctConv(
            nf, nf, group_div[2], style_channel, sk, 1, 1,
            nf, alpha_in, alpha_out)
        self.OctConv3_2 = OctConv(nf, nf // 2, 1, 1,
                                  alpha_in=alpha_in, alpha_out=alpha_out,
                                  type='last', freq_ratio=freq_ratio)
        self.conv4 = nn.Conv2d(nf // 2, out_dim, 1)

    def forward_test(self, content, style):
        out = self.AdaOctConv1_1(content, style, 'test')
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style, 'test')
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        out = self.AdaOctConv3_1(out, style, 'test')
        out = self.OctConv3_2(out)
        return self.conv4(out[0])  # combined frequency output


# ══════════════════════════════════════════════════════════════════════════════
#  Full model (test-only wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class AesFANet(nn.Module):
    """AesFA inference model: Content Encoder + Style Encoder + Decoder."""

    def __init__(self, nf=64, alpha_in=0.5, alpha_out=0.5,
                 style_kernel=3, freq_ratio=(1, 1)):
        super().__init__()
        sk = [style_kernel, style_kernel]
        self.netE = Encoder(3, nf, sk, alpha_in, alpha_out)
        self.netS = Encoder(3, nf, sk, alpha_in, alpha_out)
        self.netG = Decoder(nf, 3, 4 * nf,
                            [style_kernel, style_kernel, 3],
                            alpha_in, alpha_out, freq_ratio)

    @torch.no_grad()
    def forward(self, content, style):
        content_feat = self.netE.forward_test(content, 'content')
        style_feat = self.netS.forward_test(style, 'style')
        return self.netG.forward_test(content_feat, style_feat)
