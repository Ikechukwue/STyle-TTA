"""
xAILab Bamberg
University of Bamberg

@description:
SANET method implementation - INFERENCE ONLY
Based on: https://github.com/GlebSBrykin/SANET

Paper: "Arbitrary Style Transfer with Style-Attentional Networks"
Authors: Dae Young Park, Kwang Hee Lee
Conference: CVPR 2019

Weight preparation:
Download from the official repository:
    - vgg_normalised.pth (VGG-19 encoder)
    - decoder.pth (trained decoder)
    - transformer.pth (trained SANet transform module)
    From: https://github.com/GlebSBrykin/SANET

Merge into a single checkpoint:
python -c "
import torch, copy, torch.nn as nn
_vgg_def = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    )
_dec_def = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)

class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)

        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)

        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)

        S = torch.bmm(F, G)
        S = self.sm(S)

        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))

        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O


class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes=in_planes)
        self.sanet5_1 = SANet(in_planes=in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        out4_1 = self.sanet4_1(content4_1, style4_1)
        out5_1 = self.sanet5_1(content5_1, style5_1)
        out5_1_upsampled = self.upsample5_1(out5_1)
        merged = out4_1 + out5_1_upsampled
        output = self.merge_conv(self.merge_conv_pad(merged))
        return output

vgg_sd = torch.load('vgg_normalised.pth', weights_only=True)
_vgg_def.load_state_dict(vgg_sd)
dec_sd = torch.load('decoder_iter_500000.pth', weights_only=True)
_dec_def.load_state_dict(dec_sd)
transform = Transform(in_planes=512)
trans_sd = torch.load('transformer_iter_500000.pth', weights_only=True)
transform.load_state_dict(trans_sd)
merged = {
    'encoder': _vgg_def.state_dict(),
    'decoder': _dec_def.state_dict(),
    'transformer': transform.state_dict(),
    'config': {'method': 'sanet', 'native_size': 256},
}
torch.save(merged, 'sanet.pth')
print('Saved merged weights to sanet.pth')
"
"""

from pathlib import Path
import copy
import torch
import torch.nn as nn

from code.experiments.reference_methods.style_transfer.artistic.sanet.net import (
    SANETNet, Transform,
)


# ── VGG-19 encoder (up to relu5_1, 44 layers) ──────────────────────────────
_vgg_def = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2_2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3_4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4_4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5_4
)

# ── Decoder (mirrors VGG from relu4_1 back to RGB) ─────────────────────────
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
    SANET (Style-Attentional Network) style transfer method - INFERENCE ONLY.

    Paper: "Arbitrary Style Transfer with Style-Attentional Networks"
    Authors: Dae Young Park, Kwang Hee Lee
    Conference: CVPR 2019
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
        Default training configuration from official SANET (Train.ipynb).
        Kept for reference / reproducibility even though training is removed.
        """
        return {
            'learning_rate': 1e-4,
            'lr_decay': 5e-5,
            'max_iter': 160000,
            'batch_size': 5,
            'content_weight': 1.0,
            'style_weight': 3.0,
            'identity1_weight': 50.0,
            'identity2_weight': 1.0,
            'save_interval': 1000,
            'load_size': 512,
            'crop_size': 256,
        }

    @staticmethod
    def get_native_image_size() -> int:
        """Native resolution from official implementation (RandomCrop 256)."""
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
        Build architecture and load merged checkpoint.

        Expected checkpoint keys:
            encoder     – VGG-19 state_dict (44 layers)
            decoder     – decoder state_dict
            transformer – Transform module state_dict
        """
        checkpoint = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )

        # Build encoder
        encoder = copy.deepcopy(_vgg_def)
        encoder.load_state_dict(checkpoint['encoder'])
        encoder = nn.Sequential(*list(encoder.children())[:44])

        # Build decoder
        dec = copy.deepcopy(_dec_def)
        dec.load_state_dict(checkpoint['decoder'])

        # Assemble full network
        self.network = SANETNet(encoder, dec)

        # Load transformer weights
        self.network.transform.load_state_dict(checkpoint['transformer'])

        self.network.to(self.device)

        # Freeze everything
        for p in self.network.parameters():
            p.requires_grad = False

        self.network.eval()
