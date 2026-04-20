"""
xAILab Bamberg
University of Bamberg

@description:
Avatar-Net method implementation (inference only).
Based on: https://github.com/LucasSheng/avatar-net

Paper: "Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration"
Venue: CVPR 2018

NOTE: The original code is TensorFlow-only.  This is a faithful PyTorch
      re-implementation.  The *style decorator* (ZCA + patch swap) has NO
      learned parameters.  Only the decoder needs trained weights.

Pretrained weights:
  Original TF checkpoint:
    https://drive.google.com/open?id=1_7x93xwZMhCL-kLrz4B2iZ01Y8Q7SlTX

  The checkpoint must be converted from TF → PyTorch.  See the conversion
  script below (requires ``tensorflow`` and ``torch`` in the same env).

  VGG encoder (shared): vgg_normalised.pth

Convert TF checkpoint and merge (outline):
python -c "
import torch, numpy as np
# pip install tensorflow==1.15  (needed for TF1 checkpoint)
import tensorflow as tf

reader = tf.compat.v1.train.NewCheckpointReader('model.ckpt-120000')

# --- decoder weights ---
# The TF decoder has variables named like:
#   combined_decoder/conv4/conv4_1/weights  (H,W,C_in,C_out in TF)
#   combined_decoder/conv4/conv4_1/biases
# Map them to our AvatarNetDecoder named modules.
dec_map = {
    'conv4/conv4_1': 'conv4_1.1',   # .1 is the Conv2d in Sequential(Pad, Conv, ReLU)
    'conv3/conv3_4': 'conv3_4.1',
    'conv3/conv3_3': 'conv3_3.1',
    'conv3/conv3_2': 'conv3_2.1',
    'conv3/conv3_1': 'conv3_1.1',
    'conv2/conv2_2': 'conv2_2.1',
    'conv2/conv2_1': 'conv2_1.1',
    'conv1/conv1_2': 'conv1_2.1',
    'conv1/conv1_1': 'conv1_1.1',
    'output':        'output.1',
}
dec_sd = {}
for tf_name, pt_name in dec_map.items():
    scope = f'combined_decoder/{tf_name}'
    w = reader.get_tensor(f'{scope}/weights')   # (H,W,Cin,Cout)
    b = reader.get_tensor(f'{scope}/biases')    # (Cout,)
    # TF conv weights: (H, W, C_in, C_out) → PyTorch: (C_out, C_in, H, W)
    dec_sd[f'{pt_name}.weight'] = torch.from_numpy(w.transpose(3,2,0,1))
    dec_sd[f'{pt_name}.bias']   = torch.from_numpy(b)

enc_sd = torch.load('vgg_normalised.pth', map_location='cpu')
torch.save({
    'encoder': enc_sd,
    'decoder': dec_sd,
    'config': {
        'feature_layers': ['relu3_1', 'relu4_1'],
        'style_coding': 'ZCA',
        'patch_size': 5,
    },
}, 'avatarnet.pth')
print('Saved avatarnet.pth')
"
"""

from pathlib import Path

import torch
import torch.nn as nn

from experiments.reference_methods.style_transfer.artistic.avatarnet.net import (
    AvatarNetModel,
    AvatarNetDecoder,
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
    nn.ReLU(),  # relu4_1
)


class Method:
    """
    Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration.

    Paper:   CVPR 2018
    Authors: Lu Sheng, Ziyi Lin, Jing Shao, Xiaogang Wang

    Patch-based style transfer: ZCA-whitened content features are matched
    to nearest-neighbour style patches and re-coloured, then decoded via a
    multi-scale decoder with AdaIN fusion at intermediate VGG levels.
    """

    def __init__(self, pretrained_weights: str = None, device: str = "cpu"):
        self.network = None
        self.is_initialized = False
        self.device = device
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError(
                "pretrained_weights must point to an existing merged weights "
                "file.  See docstring for TF→PyTorch conversion instructions."
            )
        self._initialize_network(Path(pretrained_weights))

    def get_default_config(self) -> dict:
        return {
            "alpha": 1.0,
            "style_coding": "ZCA",
            "patch_size": 5,
            "feature_layers": ["relu3_1", "relu4_1"],
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 256

    # ------------------------------------------------------------------

    def __call__(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            raise RuntimeError("Network not initialised.")

        dev = content.device
        if str(self.device) != str(dev):
            self.device = dev
            self.network.to(dev)

        self.network.eval()
        with torch.no_grad():
            output = self.network(
                content, style, alpha=self.config["alpha"])
        return output.clamp(0, 1)

    # ------------------------------------------------------------------

    def _initialize_network(self, weights_path: Path):
        import copy

        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)

        cfg = ckpt.get("config", {})
        self.config["style_coding"]   = cfg.get("style_coding", self.config["style_coding"])
        self.config["patch_size"]     = cfg.get("patch_size", self.config["patch_size"])
        self.config["feature_layers"] = cfg.get("feature_layers", self.config["feature_layers"])

        # Encoder
        encoder = copy.deepcopy(_vgg_def)
        encoder.load_state_dict(ckpt["encoder"], strict=False)

        # Decoder
        decoder = AvatarNetDecoder()
        decoder.load_state_dict(ckpt["decoder"])

        self.network = AvatarNetModel(
            encoder, decoder,
            feature_layers=self.config["feature_layers"],
            style_coding=self.config["style_coding"],
            patch_size=self.config["patch_size"],
        )
        self.network.to(self.device)
        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True
