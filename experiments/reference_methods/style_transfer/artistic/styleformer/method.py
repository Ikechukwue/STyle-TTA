"""
xAILab Bamberg
University of Bamberg

@description:
StyleFormer method implementation - INFERENCE ONLY
Based on: https://github.com/Wxl-stars/PytorchStyleFormer

Paper: "StyleFormer: Real-Time Arbitrary Style Transfer via Parametric Style Composition"
Conference: ICCV 2021

Weight preparation:
Download from the official repository:
    - Official checkpoint contains keys 'a' (model state_dict) and 'b' (decoder state_dict)
    - VGG-16 weights (vgg16.pth from torchvision or official repo)
    From: https://github.com/Wxl-stars/PytorchStyleFormer

Merge into a single checkpoint:
python -c "
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std_vector = feat_var.sqrt()
    feat_mean_vector = feat.view(N, C, -1).mean(dim=2)
    return feat_mean_vector, feat_std_vector

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:11]
        self.slice3 = vgg[11:18]
        self.slice4 = vgg[18:28]

    def forward(self, x):
        out = []
        x = self.slice1(x)
        out.append(x)
        x = self.slice2(x)
        out.append(x)
        x = self.slice3(x)
        out.append(x)
        x = self.slice4(x)
        out.append(x)
        return out


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.model = nn.Sequential(
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

    def forward(self, stylized_feature):
        return self.model(stylized_feature)


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, stride=stride, bias=use_bias)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            self.activation = activation()
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None
        self.p = padding

    def forward(self, x):
        x = F.pad(x, (self.p, self.p, self.p, self.p), mode='reflect')
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Coeffs(nn.Module):
    def __init__(self, nin=16, nout=17, luma_bins=8, channel_multiplier=1,
                 spatial_bin=8, group_num=16, n_input_size=64, n_input_channel=256):
        super(Coeffs, self).__init__()
        self.nin = nin
        self.nout = nout
        self.lb = luma_bins
        self.cm = channel_multiplier
        self.sb = spatial_bin
        self.G = group_num
        bn = False
        nsize = n_input_size
        nchannel = n_input_channel
        self.relu = nn.ReLU()

        n_layers_splat = int(np.log2(nsize / self.sb))
        self.splat_features = nn.ModuleList()
        prev_ch = nchannel
        for i in range(n_layers_splat):
            self.splat_features.append(ConvBlock(prev_ch, nchannel, 3, stride=2, batch_norm=False))
            prev_ch = nchannel

        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(nchannel, 32 * self.cm * self.lb, 3, stride=1, batch_norm=bn))
        self.local_features.append(ConvBlock(32 * self.cm * self.lb, 32 * self.cm * self.lb, 3, stride=1, activation=None, use_bias=False))

        self.conv_out = ConvBlock(32 * self.cm * self.lb, self.G * self.lb * nout * nin, 1, padding=0, activation=None)

    def forward(self, lowres_input):
        bs = lowres_input.shape[0]
        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        x_local = x
        for layer in self.local_features:
            x_local = layer(x_local)
        fusion = self.relu(x_local)
        x = self.conv_out(fusion)
        s = x.shape
        x = x.view(bs * self.G, self.nin * self.nout, self.lb, s[2], s[3])
        return x


class GuideNN(nn.Module):
    def __init__(self, group_num=16):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(16, 4, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(4, 1, kernel_size=1, padding=0, activation='sigmoid')
        self.G = group_num

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, affine_transformation, guidemap):
        device = affine_transformation.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)], indexing='ij')
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)
        coeff = F.grid_sample(affine_transformation, guidemap_guide, mode='bilinear',
                              padding_mode='reflection', align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self, group_num=16, alpha=0.8, selection='Ax+b'):
        super(ApplyCoeffs, self).__init__()
        self.G = group_num
        self.alpha = alpha
        self.sect = selection

    def forward(self, coeff, full_res_input):
        N, C, H, W = full_res_input.shape
        CG = C // self.G
        output = []
        for i in range(self.G):
            if self.sect == 'Ax+b':
                x = torch.sum(full_res_input * coeff[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True) + coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            if self.sect == 'Ax':
                x = torch.sum(full_res_input * coeff[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True)
            if self.sect == 'x+b':
                x = torch.sum(full_res_input, dim=1, keepdim=True) + coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            if self.sect == 'b':
                x = coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            if self.sect == 'aAx+b':
                x = torch.sum(full_res_input * self.alpha*coeff[:, i*(CG+1):(i+1)*(CG+1)-1, :, :], dim=1, keepdim=True) + coeff[:, (i+1)*(CG+1)-1:(i+1)*(CG+1), :, :]
            output.append(x)
        return torch.cat(output, dim=1)


class AttModule(nn.Module):
    def __init__(self, luma_bins=8, group_num=16, n_input_channel=256, spatial_bin=8):
        super(AttModule, self).__init__()
        self.convc1 = ConvBlock(16, 16, stride=2)
        self.convc2 = ConvBlock(16, 16, stride=2, activation=None)
        self.convs1 = ConvBlock(16, 16, stride=2)
        self.convs2 = ConvBlock(16, 16, stride=2, activation=None)
        self.grid_channel = luma_bins * 16 * 17
        self.convsr = ConvBlock(self.grid_channel, self.grid_channel, activation=None)
        self.G = n_input_channel // group_num
        self.cpg = group_num
        self.sp = spatial_bin

    def forward(self, c, s, grid):
        Ng, Cg, Lg, Hg, Wg = grid.shape
        c = self.convc1(c)
        c1 = self.convc2(c)
        Ng, C, H, W = c1.shape
        c1 = c1.view(Ng, 16, -1)
        s = self.convs1(s)
        s1 = self.convs2(s).view(Ng, 16, -1)
        cs = torch.bmm(c1.permute(0, 2, 1), s1)
        cs = F.softmax(cs, dim=2)
        grid = grid.view(Ng, -1, Hg, Wg)
        sr = self.convsr(grid).view(Ng, self.grid_channel, -1)
        rs = torch.bmm(sr, cs.permute(0, 2, 1))
        return rs.view(Ng, Cg, Lg, H, W)


class StyleFormer(nn.Module):
    def __init__(self, luma_bins=8, channel_multiplier=1, spatial_bin=8,
                 group_num=16, n_input_size=64, n_input_channel=256,
                 alpha=0.8, selection='Ax+b'):
        super(StyleFormer, self).__init__()
        self.coeffs = Coeffs(nin=16, nout=17, luma_bins=luma_bins,
                             channel_multiplier=channel_multiplier,
                             spatial_bin=spatial_bin, group_num=group_num,
                             n_input_size=n_input_size,
                             n_input_channel=n_input_channel)
        self.att = AttModule(luma_bins=luma_bins, group_num=group_num,
                             n_input_channel=n_input_channel, spatial_bin=spatial_bin)
        self.guide = GuideNN(group_num=group_num)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs(group_num=group_num, alpha=alpha, selection=selection)
        self.G = group_num
        self.fcc = nn.Conv2d(256, 256, 1, 1)
        self.fcs = nn.Conv2d(256, 256, 1, 1)

    def forward(self, style_feat, content_feat):
        coeffs = self.coeffs(style_feat)
        content_feat = self.fcc(content_feat)
        style_feat = self.fcs(style_feat)
        content_feat = F.group_norm(content_feat, num_groups=self.G)
        style_norm = F.group_norm(style_feat, num_groups=self.G)
        N, C, H, W = content_feat.shape
        content_feat = content_feat.view(N * self.G, C // self.G, H, W)
        N, C, Hs, Ws = style_feat.shape
        style_feat = style_feat.view(N * self.G, -1, Hs, Ws)
        style_norm = style_norm.view(N * self.G, -1, Hs, Ws)
        att_coeffs = self.att(content_feat, style_norm, coeffs)
        guide = self.guide(content_feat)
        slice_coeffs = self.slice(att_coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, content_feat)
        out = out.view(N, C, H, W)
        return out


class StyleFormerNet(nn.Module):
    def __init__(self, luma_bins=8, channel_multiplier=1, spatial_bin=8,
                 group_num=16, n_input_size=64, n_input_channel=256,
                 alpha=0.8, selection='Ax+b'):
        super(StyleFormerNet, self).__init__()
        self.vgg = VGG()
        self.model = StyleFormer(
            luma_bins=luma_bins,
            channel_multiplier=channel_multiplier,
            spatial_bin=spatial_bin,
            group_num=group_num,
            n_input_size=n_input_size,
            n_input_channel=n_input_channel,
            alpha=alpha,
            selection=selection,
        )
        self.decoder = decoder()

    def forward(self, content, style):
        content_feats = self.vgg(content)
        style_feats = self.vgg(style)
        stylized_feature = self.model(style_feats[-2], content_feats[-2])  # relu3_1
        output = self.decoder(stylized_feature)
        return output

# Load official checkpoint
ckpt = torch.load('gen_00794001.pt', weights_only=False, map_location=torch.device('cpu'))
# Build network with official hyperparams

net = StyleFormerNet(
    luma_bins=4, channel_multiplier=1, spatial_bin=16,
    group_num=16, n_input_size=64, n_input_channel=256,
    alpha=1, selection='Ax+b',
)
def _strip_module(sd):
    return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
net.model.load_state_dict(_strip_module(ckpt['a']))
net.decoder.load_state_dict(_strip_module(ckpt['b']))
# Load VGG weights from torchvision format
vgg_tv = models.vgg16(pretrained=False)
vgg_tv.load_state_dict(torch.load('vgg16-397923af.pth', map_location=torch.device('cpu')))
vgg_conv_list = [2, 5, 9, 12, 16, 19, 22, 26]
vgg_model_conv_list = [0, 2, 5, 7, 10, 12, 14, 17]
vgg_layers = list(net.vgg.slice1) + list(net.vgg.slice2) + list(net.vgg.slice3) + list(net.vgg.slice4)
vgg_seq = torch.nn.Sequential(*vgg_layers)
for i in range(8):
    vgg_seq[vgg_conv_list[i]].weight = vgg_tv.features[vgg_model_conv_list[i]].weight
    vgg_seq[vgg_conv_list[i]].bias = vgg_tv.features[vgg_model_conv_list[i]].bias
merged = {
    'vgg': net.vgg.state_dict(),
    'model': net.model.state_dict(),
    'decoder': net.decoder.state_dict(),
    'config': {
       'method': 'styleformer',
       'luma_bins': 4, 'spatial_bin': 16, 'group_num': 16,
       'n_input_size': 64, 'n_input_channel': 256,
       'alpha': 1, 'selection': 'Ax+b',
    },
}
torch.save(merged, 'styleformer.pth')
print('Saved merged weights to styleformer.pth')
"
"""

from pathlib import Path
import torch

from experiments.reference_methods.style_transfer.artistic.styleformer.net import (
    StyleFormerNet,
)


class Method:
    """
    StyleFormer style transfer method - INFERENCE ONLY.

    Paper: "StyleFormer: Real-Time Arbitrary Style Transfer via Parametric Style Composition"
    Authors: Xiaolei Wu, Zhihao Hu, Lu Sheng, Dong Xu
    Conference: ICCV 2021
    """

    def __init__(
        self,
        pretrained_weights: str = None,
        device: str = 'cpu',
    ):
        self.network = None
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError(
                "pretrained_weights must point to a merged .pth file "
                "(see module docstring for merge script)"
            )

        self._initialize_network(Path(pretrained_weights))

    # ── configuration ───────────────────────────────────────────────────────
    def get_default_config(self) -> dict:
        """
        Default configuration from official StyleFormer.
        Kept for reference / reproducibility even though training is removed.
        """
        return {
            'learning_rate': 0.0001,
            'beta1': 0.8,
            'beta2': 0.999,
            'weight_decay': 0.0001,
            'amsgrad': True,
            'batch_size': 10,
            'epoch': 100,
            'content_weight': 60,
            'style_weight': 1,
            'tv_weight': 0,
            'image_size': 256,
            'save_freq': 50000,
            'luma_bins': 4,
            'channel_multiplier': 1,
            'spatial_bin': 16,
            'group_num': 16,
            'n_input_channel': 256,
            'n_input_size': 64,
            'alpha': 1,
            'selection': 'Ax+b',
        }

    @staticmethod
    def get_native_image_size() -> int:
        """Native resolution from official implementation."""
        return 256

    # ── inference ───────────────────────────────────────────────────────────
    def __call__(self, content, style):
        """
        Perform style transfer.

        Args:
            content: (B, 3, H, W) tensor in [0, 1]
            style:   (B, 3, H, W) tensor in [0, 1]

        Returns:
            Stylized image (B, 3, H, W) clamped to [0, 1]
        """
        if self.network is None:
            raise RuntimeError("Network not initialized.")

        with torch.no_grad():
            output = self.network(content, style)

        return output.clamp(0, 1)

    # ── weight loading ──────────────────────────────────────────────────────
    def _initialize_network(self, weights_path: Path):
        """
        Build architecture and load merged checkpoint.

        Expected checkpoint keys:
            vgg     – VGG encoder state_dict
            model   – StyleFormer transformer state_dict
            decoder – decoder state_dict
        """
        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        self.network = StyleFormerNet(
            luma_bins=self.config['luma_bins'],
            channel_multiplier=self.config['channel_multiplier'],
            spatial_bin=self.config['spatial_bin'],
            group_num=self.config['group_num'],
            n_input_size=self.config['n_input_size'],
            n_input_channel=self.config['n_input_channel'],
            alpha=self.config['alpha'],
            selection=self.config['selection'],
        )

        self.network.vgg.load_state_dict(checkpoint['vgg'])
        self.network.model.load_state_dict(checkpoint['model'])
        self.network.decoder.load_state_dict(checkpoint['decoder'])

        self.network.to(self.device)

        for p in self.network.parameters():
            p.requires_grad = False

        self.network.eval()
