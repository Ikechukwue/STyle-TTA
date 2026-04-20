"""
xAILab Bamberg
University of Bamberg

@description:
AdaAttN method implementation for reference methods framework (inference only).
Based on official PyTorch implementation: https://github.com/Huage001/AdaAttN

Pretrained weights download:
  1. Model: https://drive.google.com/file/d/1XvpD1eI4JeCBIaW5uwMT6ojF_qlzM_lo/view (AdaAttN_model.zip)
  2. VGG:   https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view (vgg_normalised.pth)

Merge weights into a single file:
python -c "
import torch, collections
vgg = torch.load('vgg_normalised.pth', map_location='cpu')
dec = torch.load('latest_net_decoder.pth', map_location='cpu')
trans = torch.load('latest_net_transformer.pth', map_location='cpu')
ada3 = torch.load('latest_net_adaattn_3.pth', map_location='cpu')
strip = lambda d: collections.OrderedDict((k.replace('module.','',1),v) for k,v in d.items())
torch.save({
    'encoder': vgg,
    'decoder': strip(dec),
    'transformer': strip(trans),
    'adaattn_3': strip(ada3),
    'config': {'shallow_layer': True, 'skip_connection_3': True},
}, 'adaattn.pth')
print('Saved merged weights to adaattn.pth')
"
"""

from pathlib import Path

import torch
import torch.nn as nn

from experiments.reference_methods.style_transfer.artistic.adaattn.net import (
    AdaAttNNet, vgg, Decoder, Transformer, AdaAttN, mean_variance_norm
)


class Method:
    """
    AdaAttN (Adaptive Attentional Instance Normalization) style transfer method.

    Paper: "AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer"
    Authors: Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling Wang, Xin Li,
             Zhengxing Sun, Qian Li, Errui Ding
    Conference: ICCV 2021
    """

    def __init__(
        self,
        pretrained_weights: str = None,
        device: str = "cpu",
    ):
        """
        Args:
            pretrained_weights: Path to merged weights file containing encoder,
                decoder, transformer, and (optionally) adaattn_3 state dicts.
            device: Device to load model to ('cpu' or 'cuda').
        """
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
    # Configuration helpers
    # ------------------------------------------------------------------
    def get_default_config(self) -> dict:
        """Default hyper-parameters matching the official release."""
        return {
            "load_size": 512,
            "crop_size": 256,
            "shallow_layer": True,
            "skip_connection_3": True,
            "max_sample": 64 * 64,
        }

    @staticmethod
    def get_native_image_size() -> int:
        """Native training resolution (256 for AdaAttN)."""
        return 256

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def __call__(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Perform style transfer.

        Args:
            content: (B, 3, H, W) tensor in [0, 1].
            style:   (B, 3, H, W) tensor in [0, 1].
        Returns:
            Stylized image tensor (B, 3, H, W) in [0, 1].
        """
        if self.network is None:
            raise RuntimeError("Network not initialized.")

        input_device = content.device
        if str(self.device) != str(input_device):
            self.device = input_device
            self.network.to(input_device)
            for layer in self.network.encoder_layers:
                layer.to(input_device)

        self.network.eval()
        with torch.no_grad():
            output = self.network(content, style)
        return output.clamp(0, 1)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def _initialize_network(self, weights_path: Path):
        """
        Build the AdaAttN network and load the merged pretrained weights.

        Expected keys in the checkpoint dict:
            'encoder'     - VGG-19 normalised state dict
            'decoder'     - Decoder state dict
            'transformer' - Transformer state dict
            'adaattn_3'   - (optional) AdaAttN skip-connection module
            'config'      - (optional) {'shallow_layer': bool, 'skip_connection_3': bool}
        """
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # ---- config from checkpoint (fallback to defaults) ----
        ckpt_cfg = checkpoint.get("config", {})
        shallow_layer = ckpt_cfg.get("shallow_layer", self.config["shallow_layer"])
        skip_connection_3 = ckpt_cfg.get(
            "skip_connection_3",
            "adaattn_3" in checkpoint,
        )

        # ---- encoder ----
        image_encoder = vgg
        image_encoder.load_state_dict(checkpoint["encoder"])
        image_encoder.to(self.device)

        enc_layers = list(image_encoder.children())
        enc_1 = nn.Sequential(*enc_layers[:4])
        enc_2 = nn.Sequential(*enc_layers[4:11])
        enc_3 = nn.Sequential(*enc_layers[11:18])
        enc_4 = nn.Sequential(*enc_layers[18:31])
        enc_5 = nn.Sequential(*enc_layers[31:44])
        encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]

        for layer in encoder_layers:
            for p in layer.parameters():
                p.requires_grad = False

        # ---- key_planes ----
        if shallow_layer:
            key_planes_4 = 512 + 256 + 128 + 64
        else:
            key_planes_4 = 512

        # ---- transformer ----
        transformer = Transformer(
            in_planes=512, key_planes=key_planes_4, shallow_layer=shallow_layer,
        )
        transformer.load_state_dict(checkpoint["transformer"])

        # ---- decoder ----
        decoder = Decoder(skip_connection_3=skip_connection_3)
        decoder.load_state_dict(checkpoint["decoder"])

        # ---- optional skip-connection module ----
        adaattn_3 = None
        if skip_connection_3 and "adaattn_3" in checkpoint:
            key_planes_3 = (256 + 128 + 64) if shallow_layer else 256
            adaattn_3 = AdaAttN(
                in_planes=256,
                key_planes=key_planes_3,
                max_sample=self.config["max_sample"],
            )
            adaattn_3.load_state_dict(checkpoint["adaattn_3"])

        # ---- assemble ----
        self.network = AdaAttNNet(
            encoder=encoder_layers,
            decoder=decoder,
            transformer=transformer,
            adaattn_3=adaattn_3,
            shallow_layer=shallow_layer,
        )
        self.network.to(self.device)
        for layer in self.network.encoder_layers:
            layer.to(self.device)

        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True
