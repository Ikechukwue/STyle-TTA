"""
xAILab Bamberg
University of Bamberg

@description:
CAST method implementation for reference methods framework (inference only).
Based on: https://github.com/zyxElsa/CAST_pytorch

Pretrained weights download:
  Official model from https://github.com/zyxElsa/CAST_pytorch
  Google Drive: https://drive.google.com/file/d/11dZqu95QfnAgkzgR1NTJfQutz8JlwRY8

  Download CAST_model.zip which contains:
    - vgg_normalised.pth (pretrained VGG encoder)
    - decoder_iter_160000.pth (decoder B, content->style direction)

  Note: CAST uses dual decoders (A and B) for bidirectional transfer.
  For inference we only need decoder_b (B direction).

Merge weights into a single file:
python -c "
import torch
vgg = torch.load('vgg_normalised.pth', map_location='cpu')
dec = torch.load('decoder_iter_160000.pth', map_location='cpu')
torch.save({
    'encoder': vgg,
    'decoder': dec,
}, 'cast.pth')
print('Saved merged weights to cast.pth')
"
"""

from pathlib import Path

import torch
import torch.nn as nn

from code.experiments.reference_methods.style_transfer.artistic.cast.net import (
    ADAIN_Encoder, Decoder,
)


# VGG-19 (normalised) definition - used to build encoder
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
    CAST (Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning).

    Paper: "Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning"
    Conference: SIGGRAPH 2022
    """

    def __init__(self, pretrained_weights: str = None, device: str = "cpu"):
        self.encoder = None
        self.decoder = None
        self.is_initialized = False
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError("pretrained_weights must point to an existing merged weights file.")
        self._initialize_network(Path(pretrained_weights))

    def get_default_config(self) -> dict:
        return {
            "crop_size": 256,
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
            Stylized image (B, 3, H, W) in [0, 1].
        """
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Network not initialized.")

        dev = content.device
        if str(self.device) != str(dev):
            self.device = dev
            self.encoder.to(dev)
            self.decoder.to(dev)

        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            adain_feat = self.encoder(content, style)
            output = self.decoder(adain_feat)
        return output.clamp(0, 1)

    # ------------------------------------------------------------------
    def _initialize_network(self, weights_path: Path):
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Build VGG encoder (first 31 layers up to relu4_1)
        import copy
        vgg_model = copy.deepcopy(_vgg_def)
        vgg_model.load_state_dict(checkpoint["encoder"], strict=False)
        self.encoder = ADAIN_Encoder(vgg_model)

        # Build decoder
        self.decoder = Decoder()
        self.decoder.load_state_dict(checkpoint["decoder"])

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.encoder.eval()
        self.decoder.eval()
        self.is_initialized = True
