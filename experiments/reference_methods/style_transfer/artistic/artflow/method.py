"""
xAILab Bamberg
University of Bamberg

@description:
ArtFlow method implementation for reference methods framework (inference only).
Based on: https://github.com/pkuanjie/ArtFlow

Pretrained weights download:
  Official checkpoints from https://github.com/pkuanjie/ArtFlow
  Google Drive: https://drive.google.com/drive/folders/1w2fHgSBYwjplfeCXI8eOGYpi69CpJBTE

  Each operator has its own model file:
    - model_adain.pth   (Glow + AdaIN)
    - model_wct.pth     (Glow + WCT)
    - model_decorator.pth (Glow + StyleDecorator)

  The checkpoint format is a raw state_dict.
  No merging needed - single file per operator.
"""

from pathlib import Path

import torch

from experiments.reference_methods.style_transfer.artistic.artflow.net import (
    GlowAdaIN, GlowWCT, GlowDecorator,
)


class Method:
    """
    ArtFlow (Unbiased Image Style Transfer via Reversible Neural Flows).

    Paper: "ArtFlow: Unbiased Image Style Transfer via Reversible Neural Flows"
    Conference: CVPR 2021

    Uses reversible normalizing flows (Glow) with different style transfer
    operators (AdaIN, WCT, StyleDecorator) in the latent space to avoid
    content leakage.
    """

    def __init__(
        self,
        operator: str = "adain",
        pretrained_weights: str = None,
        device: str = "cpu",
    ):
        self.operator = operator
        self.network = None
        self.is_initialized = False
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError("pretrained_weights must point to an existing weights file.")
        self._initialize_network(Path(pretrained_weights))

    def get_default_config(self) -> dict:
        return {
            "n_flow": 8,
            "n_block": 2,
            "affine": False,
            "conv_lu": True,
            "operator": "adain",
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
        if self.network is None:
            raise RuntimeError("Network not initialized.")

        dev = content.device
        if str(self.device) != str(dev):
            self.device = dev
            self.network.to(dev)

        self.network.eval()
        with torch.no_grad():
            z_c = self.network(content, forward=True)
            z_s = self.network(style, forward=True)
            output = self.network(z_c, forward=False, style=z_s)
        return output.clamp(0, 1)

    # ------------------------------------------------------------------
    def _initialize_network(self, weights_path: Path):
        cfg = self.config
        if self.operator == "adain":
            self.network = GlowAdaIN(
                3, cfg["n_flow"], cfg["n_block"],
                affine=cfg["affine"], conv_lu=cfg["conv_lu"],
            )
        elif self.operator == "wct":
            self.network = GlowWCT(
                3, cfg["n_flow"], cfg["n_block"],
                affine=cfg["affine"], conv_lu=cfg["conv_lu"],
            )
        elif self.operator == "decorator":
            self.network = GlowDecorator(
                3, cfg["n_flow"], cfg["n_block"],
                affine=cfg["affine"], conv_lu=cfg["conv_lu"],
            )
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if "state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["state_dict"])
        else:
            self.network.load_state_dict(checkpoint)

        self.network.to(self.device)
        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True
