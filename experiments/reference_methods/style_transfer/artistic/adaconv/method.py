"""
xAILab Bamberg
University of Bamberg

@description:
AdaConv method implementation - INFERENCE ONLY
Based on: https://github.com/RElbers/ada-conv-pytorch

Paper: "Adaptive Convolutions for Structure-Aware Style Transfer"
Authors: Prashanth Chandran, Gaspard Zoss, Paulo Gotardo,
         Markus Gross, Derek Bradley
Conference: CVPR 2021

NOTE: AdaConv uses ImageNet-normalized VGG-19 (torchvision) as encoder.
      Input images should be [0, 1] float tensors — normalization is
      applied internally by the VGGEncoder.

Weight preparation:
Download the pretrained model.ckpt from:
https://drive.google.com/file/d/17h-Hd08n-f_5D8cDV08dpB_-W1cs5jbt
(Official PyTorch Lightning checkpoint)

Extract AdaConvModel weights from the Lightning checkpoint:
python -c "
import torch
ckpt = torch.load('model.ckpt', map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict']
# Lightning wraps the model under 'model.' prefix
model_sd = {}
for k, v in state_dict.items():
    if k.startswith('model.'):
        model_sd[k[len('model.'):]] = v
merged = {
    'model': model_sd,
    'config': {
        'method': 'adaconv',
        'style_size': 256,
        'style_channels': 512,
        'kernel_size': 3,
    },
}
torch.save(merged, 'adaconv.pth')
"
"""

from pathlib import Path
import torch

from experiments.reference_methods.style_transfer.artistic.adaconv.net import (
    AdaConvModel,
)


class Method:
    """
    AdaConv style transfer method - INFERENCE ONLY.

    Paper: "Adaptive Convolutions for Structure-Aware Style Transfer"
    CVPR 2021
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
        Default configuration from official AdaConv implementation.
        """
        return {
            'style_size': 256,
            'style_channels': 512,
            'kernel_size': 3,
            'learning_rate': 0.0001,
            'lr_decay': 0.00005,
            'iterations': 160000,
            'style_weight': 10.0,
            'content_weight': 1.0,
            'style_loss': 'mm',  # moment matching
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
        Build AdaConvModel and load merged checkpoint.

        Expected checkpoint keys:
            model – AdaConvModel state_dict
        """
        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        self.network = AdaConvModel(
            style_size=self.config['style_size'],
            style_channels=self.config['style_channels'],
            kernel_size=self.config['kernel_size'],
        )

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.network.load_state_dict(checkpoint['model'])
        else:
            # Assume direct state_dict
            self.network.load_state_dict(checkpoint)

        self.network.to(self.device)

        for p in self.network.parameters():
            p.requires_grad = False

        self.network.eval()
