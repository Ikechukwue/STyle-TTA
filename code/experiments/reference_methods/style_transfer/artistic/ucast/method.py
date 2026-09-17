"""
UCAST – Unified Content-Aware Style Transfer (ACM TOG 2023).

Reuses the same ADAIN_Encoder + Decoder architecture as CAST, but
with weights from the unified adaptive contrastive learning training.

Official repo: https://github.com/zyxElsa/CAST_pytorch

Pretrained weights download:
  UCAST model from https://github.com/zyxElsa/CAST_pytorch
  Google Drive: https://drive.google.com/file/d/1rU8haiPG2BDhh5BNSwngjMKBKdutDYTJ

  You also need: vgg_normalised.pth (same as CAST)

Merge weights:
python -c "
import torch
vgg = torch.load('vgg_normalised.pth', map_location='cpu')
dec = torch.load('latest_net_Dec_B.pth', map_location='cpu')
torch.save({
    'encoder': vgg,
    'decoder': dec,
}, 'ucast.pth')
print('Saved merged weights to ucast.pth')
"
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn

# Reuse CAST architecture − identical encoder + decoder
from code.experiments.reference_methods.style_transfer.artistic.cast.net import (
    ADAIN_Encoder, Decoder,
)

# VGG-19 (normalised) definition – only up to relu4_1
_vgg_def = nn.Sequential(
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
    nn.ReLU(),  # relu4-1 (index 31)
)


class Method:
    """
    UCAST – A Unified Arbitrary Style Transfer Framework via
    Adaptive Contrastive Learning  (ACM TOG 2023).

    Same inference architecture as CAST (ADAIN_Encoder + Decoder)
    but trained with the UCAST adaptive contrastive objective.
    """

    def __init__(self, weights, device="cuda"):
        self.device = device
        self.cfg = self.get_default_config()
        self._initialize_network(weights)

    @staticmethod
    def get_default_config():
        return {"crop_size": 256}

    @staticmethod
    def get_native_image_size():
        return None  # fully convolutional, arbitrary resolution

    # ── network setup ──────────────────────────────────────────────────
    def _initialize_network(self, weights):
        ckpt = torch.load(weights, map_location=self.device, weights_only=False)

        vgg = copy.deepcopy(_vgg_def)
        vgg.load_state_dict(ckpt["encoder"], strict=False)
        self.encoder = ADAIN_Encoder(vgg).to(self.device).eval()

        self.decoder = Decoder()
        self.decoder.load_state_dict(ckpt["decoder"])
        self.decoder.to(self.device).eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    # ── inference ──────────────────────────────────────────────────────
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
        adain_feat = self.encoder(c, s)
        return self.decoder(adain_feat).clamp(0, 1)
