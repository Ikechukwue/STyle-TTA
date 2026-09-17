"""
xAILab Bamberg
University of Bamberg

@description:
RAST method implementation for reference methods framework (inference only).
Based on: https://github.com/YingnanMa/RAST

Paper: "RAST: Restorable Arbitrary Style Transfer via Multi-restoration"
Authors: Yingnan Ma, Xudong Li et al.
Conference: WACV 2023

Pretrained weights download:
  1. VGG encoder (vgg_normalised.pth):
     Google Drive: https://drive.google.com/file/d/
       1cI6ubAziMdOsSJZEvfofW-iCtnCmsONL/view
       (Same normalised VGG-19 as AdaIN, SANet, etc.)

  2. Decoder (decoder_iter_160000.pth):
     GitHub: https://github.com/YingnanMa/RAST/raw/main/model/decoder_iter_160000.pth

  3. Transformer (transformer_iter_160000.pth):
     GitHub: https://github.com/YingnanMa/RAST/
       raw/main/model/transformer_iter_160000.pth

Merge weights into a single file:
python -c "
import torch, torch.nn as nn, copy

# ── build dummy architectures to verify loading ──
# VGG normalised encoder (44 layers)
vgg_def = nn.Sequential(
      nn.Conv2d(3, 3, (1, 1)),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
      nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
      nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
      nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
      nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
)
# Decoder
dec_def = nn.Sequential(
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
      nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)

class SANet(nn.Module):
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

vgg_sd = torch.load('vgg_normalised.pth', weights_only=True)
vgg_def.load_state_dict(vgg_sd)

dec_sd = torch.load('decoder_iter_160000.pth', weights_only=True)
dec_def.load_state_dict(dec_sd)

transform = Transform(in_planes=512)
trans_sd = torch.load('transformer_iter_160000.pth', weights_only=True)
transform.load_state_dict(trans_sd)

merged = {
    'encoder': vgg_def.state_dict(),
    'decoder': dec_def.state_dict(),
    'transformer': transform.state_dict(),
    'config': {'method': 'rast', 'native_size': 512},
}
torch.save(merged, 'rast.pth')
print('Saved merged weights to rast.pth')
"
"""

from pathlib import Path
import copy

import torch
import torch.nn as nn

from code.experiments.reference_methods.style_transfer.artistic.rast.net import (
    RASTNet,
)


# ── VGG-19 normalised encoder (44 layers, up to relu5_1) ────────────────────
_vgg_def = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3_4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4_4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_4
)

# ── Decoder (mirrors VGG from relu4_1 back to RGB) ──────────────────────────
_dec_def = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
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


class Method:
    """RAST – Restorable Arbitrary Style Transfer (WACV 2023)."""

    def __init__(self, weights, device="cuda"):
        self.device = device
        self.cfg = self.get_default_config()
        self._initialize_network(weights)

    @staticmethod
    def get_default_config():
        return {"imsize": 512}

    @staticmethod
    def get_native_image_size():
        return None  # fully convolutional

    @torch.no_grad()
    def __call__(self, content, style):
        """
        Parameters
        ----------
        content, style : Tensor [B, 3, H, W] in [0, 1]

        Returns
        -------
        Tensor [B, 3, H, W] in [0, 1]
        """
        c = content.to(self.device)
        s = style.to(self.device)
        return self.network(c, s).clamp(0, 1)

    def _initialize_network(self, weights):
        ckpt = torch.load(weights, map_location=self.device, weights_only=False)

        encoder = copy.deepcopy(_vgg_def)
        encoder.load_state_dict(ckpt["encoder"])
        encoder = nn.Sequential(*list(encoder.children())[:44])

        dec = copy.deepcopy(_dec_def)
        dec.load_state_dict(ckpt["decoder"])

        self.network = RASTNet(encoder, dec)
        self.network.transform.load_state_dict(ckpt["transformer"])
        self.network.to(self.device).eval()

        for p in self.network.parameters():
            p.requires_grad = False
