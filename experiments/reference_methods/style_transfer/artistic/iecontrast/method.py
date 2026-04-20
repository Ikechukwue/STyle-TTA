"""
xAILab Bamberg
University of Bamberg

@description:
IEContrAST method implementation for reference methods framework (inference only).
Based on: https://github.com/HalbertCH/IEContraAST

Pretrained weights download:
  Official: https://github.com/HalbertCH/IEContraAST
  Google Drive: https://drive.google.com/file/d/11uddn7sfe8DurHMXa0_tPZkZtYmumRNH/view

  Download:
    - vgg_normalised.pth (shared VGG encoder)
    - decoder_iter_160000.pth
    - transformer_iter_160000.pth

Merge weights into a single file:
python -c "
import torch
vgg = torch.load('vgg_normalised.pth', map_location='cpu')
dec = torch.load('decoder_iter_160000.pth', map_location='cpu')
trans = torch.load('transformer_iter_160000.pth', map_location='cpu')
torch.save({
    'encoder': vgg,
    'decoder': dec,
    'transformer': trans,
}, 'iecontrast.pth')
print('Saved merged weights to iecontrast.pth')
"
"""

from pathlib import Path
import copy

import torch
import torch.nn as nn

from experiments.reference_methods.style_transfer.artistic.iecontrast.net import (
    IEContrASTNet,
)


# VGG-19 (normalised) full definition (up to relu5_1 needed since IEContrAST uses enc_5)
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
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1 (layer 44)
)

# Decoder definition
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
    IEContrAST (Internal-External Contrastive Artistic Style Transfer).

    Paper: "IEContraAST: Contrastive Learning for Arbitrary Style Transfer"
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
            style_feats = self.network.encode_with_intermediate(style)
            content_feats = self.network.encode_with_intermediate(content)
            transformed = self.network.transform(
                content_feats[3], style_feats[3],
                content_feats[4], style_feats[4],
            )
            output = self.network.decoder(transformed)
        return output.clamp(0, 1)

    # ------------------------------------------------------------------
    def _initialize_network(self, weights_path: Path):
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        encoder = copy.deepcopy(_vgg_def)
        encoder.load_state_dict(checkpoint["encoder"], strict=False)

        dec = copy.deepcopy(_dec_def)
        dec.load_state_dict(checkpoint["decoder"])

        self.network = IEContrASTNet(encoder, dec)
        self.network.transform.load_state_dict(checkpoint["transformer"])

        self.network.to(self.device)
        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True
