"""
xAILab Bamberg
University of Bamberg

@description:
EFDM method implementation for reference methods framework (inference only).
Based on: https://github.com/YBZh/EFDM/tree/main/ArbitraryStyleTransfer

Pretrained weights download:
  Official: https://github.com/YBZh/EFDM
  Weights:
    - vgg_normalised.pth (shared VGG encoder)
    - decoder_iter_160000.pth from
      https://drive.google.com/file/d/1LrjkIAN-3EEilW_KpGm2mJFnngWuPWKS

Merge weights into a single file:
python -c "
import torch, torch.nn as nn
# Build full VGG definition then truncate to first 31 layers
vgg_full = torch.load('vgg_normalised.pth', map_location='cpu')
dec = torch.load('decoder_iter_160000.pth', map_location='cpu')
torch.save({
    'encoder': vgg_full,
    'decoder': dec,
}, 'efdm.pth')
print('Saved merged weights to efdm.pth')
"
"""

from pathlib import Path

import torch
import torch.nn as nn

from code.experiments.reference_methods.style_transfer.artistic.efdm.net import (
    EFDMNet, exact_feature_distribution_matching,
)


# VGG-19 (normalised) up to relu4_1 (31 layers)
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
    nn.ReLU(),  # relu4-1 (layer 31)
)

# Decoder (mirror of VGG encoder)
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
    """
    EFDM (Exact Feature Distribution Matching for Arbitrary Style Transfer).

    Paper: "Exact Feature Distribution Matching for Arbitrary Style Transfer
            and Domain Generalization"
    Conference: CVPR 2022
    """

    def __init__(self, pretrained_weights: str = None, device: str = "cpu"):
        self.network = None
        self.is_initialized = False
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError("pretrained_weights must point to an existing merged weights file.")
        self._initialize_network(Path(pretrained_weights))

    def get_default_config(self) -> dict:
        return {"crop_size": 256}

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
            Stylized image (B, 3, H, W) in [0, 1].
        """
        if self.network is None:
            raise RuntimeError("Network not initialized.")

        dev = content.device
        if str(self.device) != str(dev):
            self.device = dev
            self.network.to(dev)

        self.network.eval()
        with torch.no_grad():
            content_feat = self.network.encode(content)
            style_feat = self.network.encode(style)
            t = exact_feature_distribution_matching(content_feat, style_feat)
            output = self.network.decoder(t)
        return output.clamp(0, 1)

    # ------------------------------------------------------------------
    def _initialize_network(self, weights_path: Path):
        import copy
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Build encoder from VGG weights
        encoder = copy.deepcopy(_vgg_def)
        encoder.load_state_dict(checkpoint["encoder"], strict=False)

        # Build decoder
        dec = copy.deepcopy(_dec_def)
        dec.load_state_dict(checkpoint["decoder"])

        self.network = EFDMNet(encoder, dec)
        self.network.to(self.device)
        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True
