"""
xAILab Bamberg
University of Bamberg

@description:
AdaIN method implementation for reference methods framework (inference only).
Based on official PyTorch implementation: https://github.com/naoto0804/pytorch-AdaIN

Pretrained weights download (from GitHub release v0.0.0):
  - decoder.pth:          https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth
  - vgg_normalised.pth:   https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth

Merge weights into a single file:
python -c "
import torch
vgg = torch.load('vgg_normalised.pth', map_location='cpu')
dec = torch.load('decoder.pth', map_location='cpu')
torch.save({'encoder': vgg, 'decoder': dec}, 'adain.pth')
print('Saved merged weights to adain.pth')
"
"""

from pathlib import Path
import copy

import torch
import torch.nn as nn

from code.experiments.reference_methods.style_transfer.artistic.adain.net import (
    AdaINNet, vgg, decoder, adaptive_instance_normalization,
)


class Method:
    """
    AdaIN (Adaptive Instance Normalization) style transfer method.

    Paper: "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
    Authors: Xun Huang, Serge Belongie
    Conference: ICCV 2017
    """

    def __init__(
        self,
        alpha: float = 1.0,
        pretrained_weights: str = None,
        device: str = "cpu",
    ):
        """
        Args:
            alpha: Style strength (0 = content only, 1 = full stylization).
            pretrained_weights: Path to merged weights file (encoder + decoder).
            device: 'cpu' or 'cuda'.
        """
        self.alpha = alpha
        self.network = None
        self.is_initialized = False
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError(
                "pretrained_weights must point to an existing merged weights file."
            )
        self._initialize_network(Path(pretrained_weights))

    # ------------------------------------------------------------------
    def get_default_config(self) -> dict:
        return {
            "alpha": 1.0,
            "content_size": 512,
            "style_size": 512,
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 256

    # ------------------------------------------------------------------
    def __call__(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        alpha: float = None,
    ) -> torch.Tensor:
        """
        Args:
            content: (B, 3, H, W) in [0, 1].
            style:   (B, 3, H, W) in [0, 1].
            alpha:   Optional style strength override.
        Returns:
            Stylized image tensor (B, 3, H, W) in [0, 1].
        """
        if self.network is None:
            raise RuntimeError("Network not initialized.")

        if alpha is None:
            alpha = self.alpha

        input_device = content.device
        if str(self.device) != str(input_device):
            self.device = input_device
            self.network.to(input_device)

        self.network.eval()
        with torch.no_grad():
            content_feat = self.network.encode(content)
            style_feat = self.network.encode(style)
            feat = adaptive_instance_normalization(content_feat, style_feat)
            feat = feat * alpha + content_feat * (1 - alpha)
            output = self.network.decoder(feat)
        return output.clamp(0, 1)

    # ------------------------------------------------------------------
    def _initialize_network(self, weights_path: Path):
        """
        Load merged pretrained weights.

        Expected checkpoint keys:
            'encoder' - full VGG-19 normalised state dict
            'decoder' - decoder state dict
        """
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        vgg_encoder = copy.deepcopy(vgg)
        decoder_net = copy.deepcopy(decoder)

        vgg_encoder.load_state_dict(checkpoint["encoder"])
        # Truncate to relu4_1 (first 31 layers) as in official test.py
        vgg_encoder = nn.Sequential(*list(vgg_encoder.children())[:31])

        decoder_net.load_state_dict(checkpoint["decoder"])

        self.network = AdaINNet(vgg_encoder, decoder_net)
        self.network.to(self.device)

        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True
