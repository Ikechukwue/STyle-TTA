"""AesFA – Method wrapper for the style_tta framework."""

import torch
from types import SimpleNamespace

from code.experiments.reference_methods.style_transfer.artistic.aesfa.net import AesFANet


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def _normalize(img, device):
    """[0,1] CHW → ImageNet-normalised CHW."""
    m = _IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    s = _IMAGENET_STD.to(device).view(1, 3, 1, 1)
    return (img - m) / s


def _denormalize(img, device):
    """ImageNet-normalised CHW → [0,1] CHW."""
    m = _IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    s = _IMAGENET_STD.to(device).view(1, 3, 1, 1)
    return (img * s + m).clamp(0, 1)


class Method:
    # ── framework interface ────────────────────────────────────────────────
    def __init__(self, weights, device="cuda"):
        self.device = device
        self.cfg = self.get_default_config()
        self._initialize_network(weights)

    @staticmethod
    def get_default_config():
        return SimpleNamespace(
            nf=64,
            alpha_in=0.5,
            alpha_out=0.5,
            style_kernel=3,
            freq_ratio=[1, 1],
        )

    @staticmethod
    def get_native_image_size():
        return 256

    # ── network setup ──────────────────────────────────────────────────────
    def _initialize_network(self, weights):
        cfg = self.cfg
        self.model = AesFANet(
            nf=cfg.nf,
            alpha_in=cfg.alpha_in,
            alpha_out=cfg.alpha_out,
            style_kernel=cfg.style_kernel,
            freq_ratio=tuple(cfg.freq_ratio),
        ).to(self.device).eval()

        ckpt = torch.load(weights, map_location=self.device)
        # The official main.pth stores keys 'netE', 'netS', 'netG'
        if 'netE' in ckpt:
            self.model.netE.load_state_dict(ckpt['netE'])
            self.model.netS.load_state_dict(ckpt['netS'])
            self.model.netG.load_state_dict(ckpt['netG'])
        else:
            # Single merged checkpoint
            self.model.load_state_dict(ckpt)

    # ── inference ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def __call__(self, content, style):
        """
        Parameters
        ----------
        content, style : Tensor [1, 3, H, W] in [0, 1]

        Returns
        -------
        Tensor [1, 3, H, W] in [0, 1]
        """
        c = _normalize(content.to(self.device), self.device)
        s = _normalize(style.to(self.device), self.device)
        out = self.model(c, s)
        return _denormalize(out, self.device)
