"""
xAILab Bamberg
University of Bamberg

@description:
LinearStyleTransfer method implementation (inference only).
Based on: https://github.com/sunshineatnoon/LinearStyleTransfer

Paper: "Learning Linear Transformations for Fast Image and Video Style Transfer"
Venue: CVPR 2019

Pretrained weights download:
  From the official repo's Google Drive links:
    relu4_1 level (default):
      vgg_r41.pth      – VGG encoder (state_dict with named keys)
      dec_r41.pth      – decoder (state_dict with named keys)
      r41.pth          – transformation matrix (MulLayer state_dict)
    relu3_1 level:
      vgg_r31.pth, dec_r31.pth, r31.pth

Merge weights into a single file (relu4_1 example):
python -c "
import torch

enc_sd = torch.load('vgg_r41.pth', map_location='cpu')
dec_sd = torch.load('dec_r41.pth', map_location='cpu')
mat_sd = torch.load('r41.pth',    map_location='cpu')

# Map named encoder keys → nn.Sequential indices
enc_map = {'conv1':'0','conv2':'2','conv3':'5','conv4':'9','conv5':'12',
            'conv6':'16','conv7':'19','conv8':'22','conv9':'25','conv10':'29'}
new_enc = {}
for k,v in enc_sd.items():
    name, param = k.split('.', 1)
    new_enc[f'{enc_map[name]}.{param}'] = v

# Map named decoder keys → nn.Sequential indices
dec_map = {'conv11':'1','conv12':'5','conv13':'8','conv14':'11',
            'conv15':'14','conv16':'18','conv17':'21','conv18':'25','conv19':'28'}
new_dec = {}
for k,v in dec_sd.items():
    name, param = k.split('.', 1)
    new_dec[f'{dec_map[name]}.{param}'] = v

torch.save({
    'encoder': new_enc,
    'decoder': new_dec,
    'matrix':  mat_sd,
    'config':  {'layer': 'r41', 'matrix_size': 32},
}, 'lst.pth')
print('Saved lst.pth')
"

For relu3_1, replace r41→r31 and use:
enc_map = {'conv1':'0','conv2':'2','conv3':'5','conv4':'9','conv5':'12','conv6':'16'}
dec_map = {'conv7':'1','conv8':'5','conv9':'8','conv10':'12','conv11':'15'}
"""

from pathlib import Path

import torch
import torch.nn as nn

from experiments.reference_methods.style_transfer.artistic.lst.net import (
    MulLayer,
    LSTNet,
)


# ── VGG-19 (normalised) encoder up to relu4_1 (31 layers) ─────────────────

_vgg_r41 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4_1
)

# VGG-19 encoder up to relu3_1 (18 layers)
_vgg_r31 = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3_1
)

# ── Decoder for relu4_1 (mirror of VGG 4→1) ───────────────────────────────

_dec_r41 = nn.Sequential(
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

# Decoder for relu3_1 (mirror of VGG 3→1)
_dec_r31 = nn.Sequential(
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
    """
    LinearStyleTransfer – Learning Linear Transformations for Fast Style Transfer.

    Paper:  CVPR 2019
    Authors: Xueting Li, Sifei Liu, Jan Kautz, Ming-Hsuan Yang

    Learns a compressed linear transformation matrix to transfer style.
    Supports relu3_1 and relu4_1 feature levels (default: relu4_1).
    """

    def __init__(self, pretrained_weights: str = None, device: str = "cpu"):
        self.network = None
        self.is_initialized = False
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError(
                "pretrained_weights must point to an existing merged weights file.  "
                "See docstring for merge instructions."
            )
        self._initialize_network(Path(pretrained_weights))

    # ------------------------------------------------------------------

    def get_default_config(self) -> dict:
        return {
            "layer": "r41",       # 'r31' or 'r41'
            "matrix_size": 32,
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 256

    # ------------------------------------------------------------------

    def __call__(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: (B, 3, H, W) in [0, 1].
            style:   (B, 3, H, W) in [0, 1].
        Returns:
            Stylised image (B, 3, H, W) in [0, 1].
        """
        if self.network is None:
            raise RuntimeError("Network not initialised.")

        dev = content.device
        if str(self.device) != str(dev):
            self.device = dev
            self.network.to(dev)

        self.network.eval()
        with torch.no_grad():
            output = self.network(content, style)
        return output.clamp(0, 1)

    # ------------------------------------------------------------------

    def _initialize_network(self, weights_path: Path):
        import copy

        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Determine level from checkpoint config or default
        cfg = ckpt.get("config", {})
        layer = cfg.get("layer", self.config["layer"])
        matrix_size = cfg.get("matrix_size", self.config["matrix_size"])
        self.config["layer"] = layer
        self.config["matrix_size"] = matrix_size

        if layer == "r31":
            in_ch = 256
            encoder = copy.deepcopy(_vgg_r31)
            decoder = copy.deepcopy(_dec_r31)
        else:  # r41
            in_ch = 512
            encoder = copy.deepcopy(_vgg_r41)
            decoder = copy.deepcopy(_dec_r41)

        encoder.load_state_dict(ckpt["encoder"], strict=False)
        decoder.load_state_dict(ckpt["decoder"])

        matrix = MulLayer(in_ch, matrix_size)
        matrix.load_state_dict(ckpt["matrix"])

        self.network = LSTNet(encoder, decoder, matrix)
        self.network.to(self.device)

        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True
