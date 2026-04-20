"""
xAILab Bamberg
University of Bamberg

@description:
AdaConv (Adaptive Convolutions) network architecture - INFERENCE ONLY
Based on: https://github.com/RElbers/ada-conv-pytorch

Paper: "Adaptive Convolutions for Structure-Aware Style Transfer"
Authors: Prashanth Chandran, Gaspard Zoss, Paulo Gotardo,
         Markus Gross, Derek Bradley
Conference: CVPR 2021
"""

import warnings
from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import transforms


# ── helpers ─────────────────────────────────────────────────────────────────

def extract_vgg_blocks(layers, layer_names):
    """Extract named blocks from VGG feature layers."""
    blocks, current_block = [], []
    scale_factor, out_channels = -1, -1
    depth_idx, relu_idx, conv_idx = 1, 1, 1
    for layer in layers:
        name = ''
        if isinstance(layer, nn.Conv2d):
            name = f'conv{depth_idx}_{conv_idx}'
            current_out_channels = layer.out_channels
            layer.padding_mode = 'reflect'
            conv_idx += 1
        elif isinstance(layer, nn.ReLU):
            name = f'relu{depth_idx}_{relu_idx}'
            layer = nn.ReLU(inplace=False)
            relu_idx += 1
        elif isinstance(layer, (nn.AvgPool2d, nn.MaxPool2d)):
            name = f'pool{depth_idx}'
            depth_idx += 1
            conv_idx = 1
            relu_idx = 1
        else:
            warnings.warn(f'Unexpected layer type: {type(layer)}')

        current_block.append(layer)
        if name in layer_names:
            blocks.append(nn.Sequential(*current_block))
            scale_factor = 1 * 2 ** (depth_idx - 1)
            out_channels = current_out_channels
            current_block = []

    return blocks, scale_factor, out_channels


# ── VGG-19 Encoder ─────────────────────────────────────────────────────────

class VGGEncoder(nn.Module):
    """
    VGG-19 encoder with ImageNet normalization.
    Extracts features at relu1_1, relu2_1, relu3_1, relu4_1.
    """

    def __init__(self):
        super().__init__()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        layer_names = {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'}
        vgg_features = models.vgg19(pretrained=False).features
        blocks, scale_factor, out_channels = extract_vgg_blocks(
            vgg_features, layer_names
        )
        self.blocks = nn.ModuleList(blocks)
        self.scale_factor = scale_factor
        self.out_channels = out_channels

    def forward(self, xs):
        xs = self.normalize(xs)
        features = []
        for block in self.blocks:
            xs = block(xs)
            features.append(xs)
        return features

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False


# ── Kernel Predictor ────────────────────────────────────────────────────────

class KernelPredictor(nn.Module):
    """Predicts adaptive convolution kernels from style embedding."""

    def __init__(self, in_channels, out_channels, n_groups, style_channels,
                 kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_channels = style_channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(
            style_channels,
            in_channels * out_channels // n_groups,
            kernel_size=kernel_size,
            padding=(ceil(padding), ceil(padding)),
            padding_mode='reflect',
        )
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1),
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels, out_channels, kernel_size=1),
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(
            len(w), self.out_channels,
            self.in_channels // self.n_groups,
            self.kernel_size, self.kernel_size,
        )
        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(
            len(w), self.out_channels,
            self.out_channels // self.n_groups, 1, 1,
        )
        bias = self.bias(w)
        bias = bias.reshape(len(w), self.out_channels)
        return w_spatial, w_pointwise, bias


# ── Adaptive Convolution ────────────────────────────────────────────────────

class AdaConv2d(nn.Module):
    """Depthwise separable adaptive convolution."""

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super().__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(ceil(padding), floor(padding)),
            padding_mode='reflect',
        )

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)
        ys = []
        for i in range(len(x)):
            y = self._forward_single(x[i:i + 1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)
        ys = self.conv(ys)
        return ys

    def _forward_single(self, x, w_spatial, w_pointwise, bias):
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))
        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x


# ── Global Style Encoder ───────────────────────────────────────────────────

class GlobalStyleEncoder(nn.Module):
    """Encodes VGG style features into a compact style descriptor."""

    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        channels = in_shape[0]

        self.downscale = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(),
            nn.AvgPool2d(2, 2),
        )

        in_features = self.in_shape[0] * (self.in_shape[1] // 8) * (self.in_shape[2] // 8)
        out_features = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, xs):
        ys = self.downscale(xs)
        ys = ys.reshape(len(xs), -1)
        w = self.fc(ys)
        w = w.reshape(len(xs), self.out_shape[0], self.out_shape[1], self.out_shape[2])
        return w


# ── AdaConv Decoder ─────────────────────────────────────────────────────────

class AdaConvDecoder(nn.Module):
    """Inverted VGG with first conv replaced by AdaConv at each scale."""

    def __init__(self, style_channels, kernel_size):
        super().__init__()
        self.style_channels = style_channels
        self.kernel_size = kernel_size

        group_div = [1, 2, 4, 8]
        n_convs = [1, 4, 2, 2]
        self.layers = nn.ModuleList([
            *self._make_layers(512, 256, group_div=group_div[0], n_convs=n_convs[0]),
            *self._make_layers(256, 128, group_div=group_div[1], n_convs=n_convs[1]),
            *self._make_layers(128, 64, group_div=group_div[2], n_convs=n_convs[2]),
            *self._make_layers(64, 3, group_div=group_div[3], n_convs=n_convs[3],
                               final_act=False, upsample=False),
        ])

    def forward(self, content, w_style):
        for module in self.layers:
            if isinstance(module, KernelPredictor):
                w_spatial, w_pointwise, bias = module(w_style)
            elif isinstance(module, AdaConv2d):
                content = module(content, w_spatial, w_pointwise, bias)
            else:
                content = module(content)
        return content

    def _make_layers(self, in_channels, out_channels, group_div, n_convs,
                     final_act=True, upsample=True):
        n_groups = in_channels // group_div
        layers = []
        for i in range(n_convs):
            last = i == n_convs - 1
            out_channels_ = out_channels if last else in_channels
            if i == 0:
                layers += [
                    KernelPredictor(in_channels, in_channels,
                                    n_groups=n_groups,
                                    style_channels=self.style_channels,
                                    kernel_size=self.kernel_size),
                    AdaConv2d(in_channels, out_channels_, n_groups=n_groups),
                ]
            else:
                layers.append(nn.Conv2d(in_channels, out_channels_, 3,
                                        padding=1, padding_mode='reflect'))
            if not last or final_act:
                layers.append(nn.ReLU())
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        return layers


# ── Main Network ────────────────────────────────────────────────────────────

class AdaConvModel(nn.Module):
    """
    Complete AdaConv network for inference.
    VGG encoder + GlobalStyleEncoder + AdaConvDecoder.
    """

    def __init__(self, style_size=256, style_channels=512, kernel_size=3):
        super().__init__()
        self.encoder = VGGEncoder()

        style_in_shape = (
            self.encoder.out_channels,
            style_size // self.encoder.scale_factor,
            style_size // self.encoder.scale_factor,
        )
        style_out_shape = (style_channels, kernel_size, kernel_size)
        self.style_encoder = GlobalStyleEncoder(
            in_shape=style_in_shape, out_shape=style_out_shape
        )
        self.decoder = AdaConvDecoder(
            style_channels=style_channels, kernel_size=kernel_size
        )

    def forward(self, content, style):
        self.encoder.freeze()
        content_embeddings = self.encoder(content)
        style_embeddings = self.encoder(style)
        style_embedding = self.style_encoder(style_embeddings[-1])
        output = self.decoder(content_embeddings[-1], style_embedding)
        return output
