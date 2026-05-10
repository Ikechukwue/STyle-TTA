"""
ArtFlow Network Architecture (inference only)

xAILab Bamberg, University of Bamberg
Based on: https://github.com/pkuanjie/ArtFlow

This file contains the network architecture for ArtFlow:
- Glow backbone (normalizing flows)
- AdaIN operator
- WCT operator
- StyleDecorator operator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg as la


logabs = lambda x: torch.log(torch.abs(x))


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and standard deviation of features."""
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


# ============================================================================
# Glow Components
# ============================================================================

class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        return F.conv2d(input, self.weight)

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        weight = np.random.randn(in_channel, out_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer('w_p', torch.from_numpy(w_p))
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(torch.from_numpy(w_s)))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(torch.from_numpy(w_l))
        self.w_s = nn.Parameter(logabs(torch.from_numpy(w_s)))
        self.w_u = nn.Parameter(torch.from_numpy(w_u))

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def forward(self, input):
        return F.conv2d(input, self.calc_weight())

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        return out * torch.exp(self.scale * 3)


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()
        self.affine = affine
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
        else:
            out_b = in_b + self.net(in_a)
        return torch.cat([in_a, out_b], 1)

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            in_b = out_b - self.net(out_a)
        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, use_coupling=True, affine=True, conv_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel) if conv_lu else InvConv2d(in_channel)
        self.use_coupling = use_coupling
        if self.use_coupling:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        input = self.actnorm(input)
        input = self.invconv(input)
        if self.use_coupling:
            input = self.coupling(input)
        return input

    def reverse(self, input):
        if self.use_coupling:
            input = self.coupling.reverse(input)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True):
        super().__init__()
        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

    def forward(self, input):
        b, c, h, w = input.shape
        squeezed = input.view(b, c, h // 2, 2, w // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b, c * 4, h // 2, w // 2)
        for flow in self.flows:
            out = flow(out)
        return out

    def reverse(self, output, reconstruct=False):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        b, c, h, w = input.shape
        unsqueezed = input.view(b, c // 4, 2, 2, h, w)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        return unsqueezed.contiguous().view(b, c // 4, h * 2, w * 2)


# ============================================================================
# Style Transfer Operators
# ============================================================================

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        assert content.size()[:2] == style.size()[:2]
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized * style_std.expand(size) + style_mean.expand(size)


def wct_core(cont_feat, styl_feat):
    """Whitening and Coloring Transform core."""
    cFSize = cont_feat.size()
    c_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat = cont_feat - c_mean

    iden = torch.eye(cFSize[0], device=cont_feat.device)
    contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1)
    c_u, c_e, c_v = torch.svd(contentConv, some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0] - 1, -1, -1):
        if c_e[i] >= 0.00001:
            k_c = i + 1
            break

    sFSize = styl_feat.size()
    s_mean = torch.mean(styl_feat, 1)
    styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
    styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
    s_u, s_e, s_v = torch.svd(styleConv, some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0] - 1, -1, -1):
        if s_e[i] >= 0.00001:
            k_s = i + 1
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, c_v[:, 0:k_c].t())
    whiten_cF = torch.mm(step2, cont_feat)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(
        torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), s_v[:, 0:k_s].t()),
        whiten_cF,
    )
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature


def covsqrt_mean(feature, inverse=False, tolerance=1e-14):
    b, c, h, w = feature.size()
    mean = torch.mean(feature.view(b, c, -1), dim=2, keepdim=True)
    zeromean = feature.view(b, c, -1) - mean
    cov = torch.bmm(zeromean, zeromean.transpose(1, 2))
    evals, evects = evals, evects = torch.linalg.eigh(cov)
    p = -0.5 if inverse else 0.5
    covsqrt = []
    for i in range(b):
        k = 0
        for j in range(c):
            if evals[i][j] > tolerance:
                k = j
                break
        covsqrt.append(
            torch.mm(
                evects[i][:, k:],
                torch.mm(evals[i][k:].pow(p).diag_embed(), evects[i][:, k:].t()),
            ).unsqueeze(0)
        )
    return torch.cat(covsqrt, dim=0), mean


def whitening(feature):
    b, c, h, w = feature.size()
    inv_covsqrt, mean = covsqrt_mean(feature, inverse=True)
    return torch.matmul(inv_covsqrt, feature.view(b, c, -1) - mean).view(b, c, h, w)


def coloring(feature, target):
    b, c, h, w = feature.size()
    covsqrt, mean = covsqrt_mean(target)
    return (torch.matmul(covsqrt, feature.view(b, c, -1)) + mean).view(b, c, h, w)


def extract_patches(feature, patch_size, stride):
    ph, pw = patch_size
    sh, sw = stride
    padh = (ph - 1) // 2
    padw = (pw - 1) // 2
    feature = F.pad(feature, (padw, padw, padh, padh), 'constant', 0)
    patches = feature.unfold(2, ph, sh).unfold(3, pw, sw)
    return patches.contiguous().view(*patches.size()[:-2], -1)


class StyleDecorator(nn.Module):
    """Patch-based style decorator operator."""

    def __init__(self):
        super().__init__()

    def kernel_normalize(self, kernel, k=3):
        b, ch, h, w, kk = kernel.size()
        kernel = kernel.view(b, ch, h * w, kk).transpose(2, 1)
        kernel_norm = torch.norm(kernel.contiguous().view(b, h * w, ch * kk), p=2, dim=2, keepdim=True)
        kernel = kernel.view(b, h * w, ch, k, k)
        kernel_norm = kernel_norm.view(b, h * w, 1, 1, 1)
        return kernel, kernel_norm

    def conv2d_with_style_kernels(self, features, kernels, patch_size, deconv_flag=False):
        output = []
        pad = (patch_size - 1) // 2
        for feature, kernel in zip(features, kernels):
            feature = F.pad(feature.unsqueeze(0), (pad, pad, pad, pad), 'constant', 0)
            if deconv_flag:
                output.append(F.conv_transpose2d(feature, kernel, padding=patch_size - 1))
            else:
                output.append(F.conv2d(feature, kernel))
        return torch.cat(output, dim=0)

    def binarize_patch_score(self, features):
        outputs = []
        for feature in features:
            matching_indices = torch.argmax(feature, dim=0)
            one_hot_mask = torch.zeros_like(feature)
            h, w = matching_indices.size()
            for i in range(h):
                for j in range(w):
                    one_hot_mask[matching_indices[i, j], i, j] = 1
            outputs.append(one_hot_mask.unsqueeze(0))
        return torch.cat(outputs, dim=0)

    def norm_deconvolution(self, h, w, patch_size):
        mask = torch.ones((h, w))
        fullmask = torch.zeros((h + patch_size - 1, w + patch_size - 1))
        for i in range(patch_size):
            for j in range(patch_size):
                pad = (i, patch_size - i - 1, j, patch_size - j - 1)
                fullmask += F.pad(mask, pad, 'constant', 0)
        pad_width = (patch_size - 1) // 2
        if pad_width == 0:
            deconv_norm = fullmask
        else:
            deconv_norm = fullmask[pad_width:-pad_width, pad_width:-pad_width]
        return deconv_norm.view(1, 1, h, w)

    def reassemble_feature(self, norm_c, norm_s, patch_size, patch_stride):
        style_kernel = extract_patches(norm_s, [patch_size, patch_size], [patch_stride, patch_stride])
        style_kernel, kernel_norm = self.kernel_normalize(style_kernel, patch_size)
        patch_score = self.conv2d_with_style_kernels(norm_c, style_kernel / kernel_norm, patch_size)
        binarized = self.binarize_patch_score(patch_score)
        deconv_norm = self.norm_deconvolution(binarized.size(2), binarized.size(3), patch_size)
        output = self.conv2d_with_style_kernels(binarized, style_kernel, patch_size, deconv_flag=True)
        return output / deconv_norm.type_as(output)

    def forward(self, content_feature, style_feature, style_strength=1.0, patch_size=3, patch_stride=1):
        norm_c = whitening(content_feature)
        norm_s = whitening(style_feature)
        reassembled = self.reassemble_feature(norm_c, norm_s, patch_size, patch_stride)
        stylized = coloring(reassembled, style_feature)
        return stylized * style_strength + content_feature * (1 - style_strength)


# ============================================================================
# Complete Glow Models
# ============================================================================

class GlowAdaIN(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4
        self.blocks.append(Block(n_channel, n_flow, affine=affine))
        self.adain = AdaIN()

    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward(input, style)
        return self._reverse(input, style)

    def _forward(self, input, style=None):
        z = input
        for block in self.blocks:
            z = block(z)
        if style is not None:
            z = self.adain(z, style)
        return z

    def _reverse(self, z, style=None):
        if style is not None:
            z = self.adain(z, style)
        for block in self.blocks[::-1]:
            z = block.reverse(z)
        return z


class GlowWCT(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4
        self.blocks.append(Block(n_channel, n_flow, affine=affine))

    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward(input, style)
        return self._reverse(input, style)

    def _forward(self, input, style=None):
        z = input
        for block in self.blocks:
            z = block(z)
        if style is not None:
            z_wct = wct_core(z.view(z.shape[1], -1), style.view(style.shape[1], -1))
            z = z_wct.view_as(z)
        return z

    def _reverse(self, z, style=None):
        if style is not None:
            z_wct = wct_core(z.view(z.shape[1], -1), style.view(style.shape[1], -1))
            z = z_wct.view_as(z)
        for block in self.blocks[::-1]:
            z = block.reverse(z)
        return z


class GlowDecorator(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 4
        self.blocks.append(Block(n_channel, n_flow, affine=affine))
        self.decorator = StyleDecorator()

    def forward(self, input, forward=True, style=None):
        if forward:
            return self._forward(input, style)
        return self._reverse(input, style)

    def _forward(self, input, style=None):
        z = input
        for block in self.blocks:
            z = block(z)
        if style is not None:
            z = self.decorator(z, style)
        return z

    def _reverse(self, z, style=None):
        if style is not None:
            z = self.decorator(z, style)
        for block in self.blocks[::-1]:
            z = block.reverse(z)
        return z
