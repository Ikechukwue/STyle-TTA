"""
xAILab Bamberg
University of Bamberg

@description:
PhotoWCT method implementation for reference methods framework (inference only).
Based on: "A Closed-form Solution to Photorealistic Image Stylization" (ECCV 2018)
Official repo: https://github.com/NVIDIA/FastPhotoStyle

Authors: Yijun Li, Ming-Yu Liu, Xueting Li, Ming-Hsuan Yang, Jan Kautz

Architecture overview
---------------------
The method uses 4 levels of VGG-19 (normalised) encoder slices paired with
4 reconstruction decoders that use MaxUnpool2d for faithful spatial up-sampling.

At inference time a coarse-to-fine cascade is run:
  encode(content, L4) → WCT → decode4 → encode(result, L3) → WCT → ... → out

The deepest‐level encoder extracts style features at all 4 levels in a single
``forward_multiple`` pass.  Each WCT step aligns content feature statistics to
the style via ZCA whitening + coloring.

A photorealistic smoothing step (Guided Image Filtering or Laplacian-based
propagation) is optionally applied as post-processing to enforce local
consistency with the content image structure.

Pretrained weights
------------------
The official repo provides a single merged checkpoint: ``photo_wct.pth``
  (in PhotoWCTModels/ on the GitHub repo, or downloadable via download_models.py)

It can also be built from the 8 individual .pth files:
  vgg_normalised_conv{1..4}.pth   – VGG encoder truncated at each level
  feature_invertor_conv{1..4}.pth – reconstruction decoders for each level

The merged ``photo_wct.pth`` state-dict has keys like:
  e1.conv0.weight, e1.conv1_1.weight, …
  d1.conv1_1.weight, d1.conv1_1.bias, …
  e2.conv0.weight, e2.conv1_1.weight, e2.conv1_2.weight, …
  d2.conv2_1.weight, …, d2.conv1_1.weight, …
  ... (up to e4, d4)

This file can be loaded directly via ``PhotoWCTNet().load_state_dict(...)``.

Alternatively, create a single file with separate sub-keys::
python -c "
import torch
# From individual .pth files (see converter.py in official repo)
photo_wct = {}
for level in [1, 2, 3, 4]:
    enc = torch.load(f'pth_models/vgg_normalised_conv{level}.pth', map_location='cpu')
    dec = torch.load(f'pth_models/feature_invertor_conv{level}.pth', map_location='cpu')
    for k, v in enc.items():
        photo_wct[f'e{level}.{k}'] = v
    for k, v in dec.items():
        photo_wct[f'd{level}.{k}'] = v
torch.save(photo_wct, 'photowct.pth')
print('Saved merged photowct.pth')
"
"""

from pathlib import Path

import torch
import numpy as np
from PIL import Image

from experiments.reference_methods.style_transfer.photorealistic.photowct.net import (
    PhotoWCTNet,
)
from experiments.reference_methods.style_transfer.photorealistic.photowct.smoothing import (
    GIFSmoothing,
)


class Method:
    """
    PhotoWCT – A Closed-form Solution to Photorealistic Image Stylization.

    Paper:      "A Closed-form Solution to Photorealistic Image Stylization"
    Conference: ECCV 2018
    Authors:    Yijun Li, Ming-Yu Liu, Xueting Li, Ming-Hsuan Yang, Jan Kautz
    Repo:       https://github.com/NVIDIA/FastPhotoStyle

    Uses 4-level VGG encoder/decoder pairs with WCT at each level and
    MaxUnpool-based decoding for spatial fidelity.  Optional GIF smoothing.
    """

    def __init__(self, pretrained_weights: str = None, device: str = "cpu"):
        self.network = None
        self.is_initialized = False
        self.device = device
        self.config = self.get_default_config()
        self.smoother = None

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError(
                "pretrained_weights must point to an existing weights file "
                "(photowct.pth).  See docstring for download instructions."
            )
        self._initialize_network(Path(pretrained_weights))

    # ------------------------------------------------------------------

    def get_default_config(self) -> dict:
        return {
            "alpha": 1.0,          # style strength (0 = content, 1 = full)
            "post_smoothing": True, # apply Guided Image Filtering
            "gif_radius": 35,       # GIF radius
            "gif_eps": 0.001,       # GIF epsilon
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 512

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
        B = content.size(0)
        results = []
        with torch.no_grad():
            for i in range(B):
                c = content[i : i + 1]
                s = style[i : i + 1]
                out = self.network(c, s, alpha=self.config["alpha"])

                # Optional photorealistic smoothing (per-image, on CPU)
                if self.config["post_smoothing"] and self.smoother is not None:
                    out_np = (
                        out.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    )
                    cont_np = (
                        c.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                    )
                    out_pil = Image.fromarray(
                        (out_np * 255).astype(np.uint8)
                    )
                    cont_pil = Image.fromarray(
                        (cont_np * 255).astype(np.uint8)
                    )
                    smoothed = self.smoother.process(out_pil, cont_pil)
                    smoothed_t = (
                        torch.from_numpy(
                            np.asarray(smoothed).astype(np.float32) / 255.0
                        )
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(dev)
                    )
                    results.append(smoothed_t)
                else:
                    results.append(out)

        return torch.cat(results, dim=0).clamp(0, 1)

    # ------------------------------------------------------------------

    def _initialize_network(self, weights_path: Path):
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)

        self.network = PhotoWCTNet()
        self.network.load_state_dict(ckpt)

        self.network.to(self.device)
        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True

        # Initialise smoother
        try:
            self.smoother = GIFSmoothing(
                r=self.config["gif_radius"],
                eps=self.config["gif_eps"],
            )
        except Exception:
            self.smoother = None
