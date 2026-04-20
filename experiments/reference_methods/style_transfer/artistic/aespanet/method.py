"""
xAILab Bamberg
University of Bamberg

@description:
AesPA-Net method implementation for reference methods framework (inference only).
Based on: https://github.com/Kibeom-Hong/AesPA-Net

Pretrained weights download:
  1. VGG:         https://drive.google.com/drive/folders/1HsJNskEMC5HUimq6ixkSZk7W_hgFNp7J (vgg_normalised_conv5_1.t7)
  2. Decoder:     https://drive.google.com/file/d/1nb7dQwj7RcQpi8_cURvErSwA-BxyZTT5/view  (dec_model.pth)
  3. Transformer: https://drive.google.com/file/d/1YII45EfR3mVbyvqQlzvfiYFIoTCgGG_R/view  (transformer_model.pth)

  Note: The VGG weights can equivalently come from the vgg_normalised.pth used
  by other methods (AdaIN, etc.) - they share the same numeric-key format.

Merge weights into a single file:
python -c "
import torch
# VGG: use the .pth format (numeric keys like '0.weight', '2.weight'...)
vgg = torch.load('vgg_normalised.pth', map_location='cpu')
dec = torch.load('dec_model.pth', map_location='cpu')['state_dict']
trans = torch.load('transformer_model.pth', map_location='cpu')['state_dict']
torch.save({
    'encoder': vgg,
    'decoder': dec,
    'transformer': trans,
}, 'aespanet.pth')
print('Saved merged weights to aespanet.pth')
"
"""

from pathlib import Path
import random
from itertools import combinations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from experiments.reference_methods.style_transfer.artistic.aespanet.net import (
    AesPANet, gram_matrix,
)


class Method:
    """
    AesPA-Net (Aesthetic Pattern-Aware Style Transfer Networks).

    Paper: "AesPA-Net: Aesthetic Pattern-Aware Style Transfer Networks"
    Conference: ICCV 2023
    """

    def __init__(self, pretrained_weights: str = None, device: str = "cpu"):
        self.network = None
        self.device = device
        self.is_initialized = False
        self.config = self.get_default_config()

        if pretrained_weights is None or not Path(pretrained_weights).exists():
            raise ValueError("pretrained_weights must point to an existing merged weights file.")
        self._initialize_network(Path(pretrained_weights))

    def get_default_config(self) -> dict:
        return {
            "imsize": 256,
            "adaptive_gram_levels": [1, 2, 3],
            "adaptive_gram_sampling": 0.05,
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
            gray_content = TF.rgb_to_grayscale(content).repeat(1, 3, 1, 1)
            gray_style = TF.rgb_to_grayscale(style).repeat(1, 3, 1, 1)

            # Official test uses only ratio=8 for adaptive alpha
            style_alpha = self._compute_pattern_repeatability(style, gray_style)

            # Official test passes `style` (not gray_style) as 5th arg
            output, _, _, _, _ = self.network(
                content, style, style_alpha, gray_content, style
            )
        return output.clamp(0, 1)

    # ------------------------------------------------------------------
    def _initialize_network(self, weights_path: Path):
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        self.network = AesPANet(encoder_state_dict=checkpoint["encoder"])

        self.network.decoder.load_state_dict(checkpoint["decoder"])
        self.network.transformer.load_state_dict(checkpoint["transformer"])

        self.network.to(self.device)
        for p in self.network.parameters():
            p.requires_grad = False
        self.network.eval()
        self.is_initialized = True

    # ------------------------------------------------------------------
    # Pattern repeatability (official: baseline.py calc_adaptive_alpha + adaptive_gram_weight)
    # ------------------------------------------------------------------
    def _compute_pattern_repeatability(self, style, gray_style):
        """Compute adaptive alpha using only ratio=8 as in official test code."""
        alphas = []
        for level in self.config["adaptive_gram_levels"]:
            a_rgb = self._adaptive_gram_weight(style, level, 8)
            a_gray = self._adaptive_gram_weight(gray_style, level, 8)
            alphas.append((a_rgb + a_gray) / 2)
        final = sum(alphas) / len(alphas)
        return final.unsqueeze(1)

    def _adaptive_gram_weight(self, image, level, ratio):
        encoded = self.network.encoder.get_features(image, level)
        global_gram = gram_matrix(encoded)
        B, C, w, h = encoded.size()
        tw, th = w // ratio, h // ratio
        if tw < 1 or th < 1:
            return torch.ones(B, device=image.device)

        patches = self._extract_patches(encoded, tw, tw)
        _, n_patches, _, _, _ = patches.size()
        if n_patches < 2:
            return torch.ones(B, device=image.device)

        cos = torch.nn.CosineSimilarity(eps=1e-6)
        comb = list(combinations(range(n_patches), 2))
        sampling_num = int(len(comb) * self.config["adaptive_gram_sampling"]) if n_patches >= 10 else len(comb)

        intra_list, inter_list = [], []
        for idx in range(B):
            # intra: patch vs global gram
            intra = []
            for p in range(n_patches):
                pg = gram_matrix(patches[idx][p].unsqueeze(0))
                intra.append(cos(global_gram[idx:idx+1], pg).mean().item())
            intra_list.append(torch.tensor(intra, device=image.device))

            # inter: pairwise patch gram
            inter = []
            for pair in random.choices(comb, k=sampling_num):
                g1 = gram_matrix(patches[idx][pair[0]].unsqueeze(0))
                g2 = gram_matrix(patches[idx][pair[1]].unsqueeze(0))
                inter.append(cos(g1, g2).mean().item())
            inter_list.append(torch.tensor(inter, device=image.device))

        intra_stat = torch.stack(intra_list).mean(dim=1)
        inter_stat = torch.stack(inter_list).mean(dim=1)
        results = (intra_stat + inter_stat) / 2

        # Official boosting: sigmoid( 10*(x - 0.6) )
        results = 1.0 / (1.0 + torch.exp(-10.0 * (results - 0.6)))
        return results

    @staticmethod
    def _extract_patches(x, kernel, stride=None):
        if stride is None:
            stride = kernel
        b, c, h, w = x.shape
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.contiguous().view(b, c, -1, kernel, kernel)
        return patches.permute(0, 2, 1, 3, 4).contiguous()
