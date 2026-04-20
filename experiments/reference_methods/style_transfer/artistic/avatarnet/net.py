"""
Avatar-Net: Multi-scale Zero-shot Style Transfer by Feature Decoration
(Sheng et al., CVPR 2018).

PyTorch re-implementation from the official TensorFlow code:
    https://github.com/LucasSheng/avatar-net

The method consists of three stages:
  1. VGG-19 encoder extracts multi-scale features.
  2. *Style decorator* at the deepest feature level: ZCA-whitens content and
     style features, swaps each content patch with its nearest style patch,
     then ZCA-re-colours with the style statistics.
  3. A trained decoder reconstructs the image from the decorated features,
     fusing intermediate-level style information via AdaIN at each scale.

NOTE: The style decorator (ZCA + nearest patch swap) has NO learned parameters.
      Only the decoder requires trained weights (for image reconstruction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ZCA whitening / colouring ──────────────────────────────────────────────

def zca_normalize(features):
    """ZCA-whiten a feature tensor.

    Args:
        features: (B, C, H, W).
    Returns:
        whitened:      (B, C, H, W) normalised features.
        color_kernel:  (B, C, C) coloring matrix (U diag(√S) U^T).
        mean:          (B, C, 1)  channel-wise mean.
    """
    B, C, H, W = features.shape
    feat = features.reshape(B, C, -1)                        # (B, C, N)
    mean = feat.mean(dim=2, keepdim=True)                    # (B, C, 1)
    centered = feat - mean

    cov = torch.bmm(centered, centered.transpose(1, 2)) / (H * W)
    u, s, _ = torch.linalg.svd(cov)                         # u: (B,C,C), s: (B,C)

    valid = (s > 1e-5).float()
    s_inv_sqrt = (1.0 / s.clamp(min=1e-5)).sqrt() * valid
    s_sqrt     = s.clamp(min=1e-5).sqrt() * valid

    # Whitening matrix: W = U diag(1/√S) U^T
    whiten_mat = u @ torch.diag_embed(s_inv_sqrt) @ u.transpose(1, 2)
    whitened   = torch.bmm(whiten_mat, centered).reshape(B, C, H, W)

    # Coloring kernel for later re-colouring
    color_kernel = u @ torch.diag_embed(s_sqrt) @ u.transpose(1, 2)

    return whitened, color_kernel, mean


def zca_colorize(whitened, color_kernel, mean):
    """Re-colour whitened features with style statistics."""
    B, C, H, W = whitened.shape
    feat = whitened.reshape(B, C, -1)
    colored = torch.bmm(color_kernel, feat) + mean
    return colored.reshape(B, C, H, W)


# ── AdaIN normalisation (alternative to ZCA) ──────────────────────────────

def adain_normalize(features):
    mean = features.mean(dim=[2, 3], keepdim=True)
    var  = features.var(dim=[2, 3], keepdim=True)
    normalized = (features - mean) / (var + 1e-7).sqrt()
    return normalized, var, mean


def adain_colorize(normalized, var, mean):
    return var.sqrt() * normalized + mean


# ── Nearest-patch swapping ─────────────────────────────────────────────────

def nearest_patch_swap(cF, sF, patch_size=5):
    """For each content patch, swap with the nearest style patch.

    Uses normalised cross-correlation for matching.
    Overlapping patches are averaged (F.fold).
    Assumes batch_size == 1.
    """
    B, C, cH, cW = cF.shape
    pad = patch_size // 2

    # Style patches (zero-padded SAME, like TF)
    s_unfold = F.unfold(sF, patch_size, padding=pad)      # (1, C*ps², N_s)
    N_s = s_unfold.size(2)

    # Conv filters from style patches: (N_s, C, ps, ps)
    filters = s_unfold[0].t().reshape(N_s, C, patch_size, patch_size)
    f_norm = filters.reshape(N_s, -1).norm(dim=1, keepdim=True).reshape(N_s, 1, 1, 1)
    f_normalised = filters / (f_norm + 1e-7)

    # Cross-correlation (reflect-pad content, VALID conv)
    cF_padded = F.pad(cF, [pad] * 4, mode='reflect')
    scores = F.conv2d(cF_padded, f_normalised)             # (1, N_s, cH, cW)

    # Best match per spatial position
    best = scores[0].argmax(dim=0).reshape(-1)              # (cH*cW,)

    # Gather matched patches and fold back
    matched = s_unfold[0, :, best].unsqueeze(0)             # (1, C*ps², cH*cW)
    output = F.fold(matched, (cH, cW), patch_size, padding=pad)

    # Overlap normalisation
    ones = torch.ones_like(matched)
    count = F.fold(ones, (cH, cW), patch_size, padding=pad)
    return output / (count + 1e-7)


# ── Style decorator ───────────────────────────────────────────────────────

def style_decorator(content_feat, style_feat,
                    coding='ZCA', patch_size=5, alpha=1.0):
    """Feature decoration: project → patch-swap → reconstruct.

    Args:
        content_feat, style_feat: (1, C, H, W) feature tensors.
        coding:     'ZCA' or 'AdaIN'.
        patch_size: neighbourhood size for patch matching.
        alpha:      interpolation ratio (1 = full style, 0 = content).
    Returns:
        Decorated features (1, C, H, W).
    """
    # ─ project ─
    if coding == 'ZCA':
        c_proj, _, _            = zca_normalize(content_feat)
        s_proj, s_kernel, s_mean = zca_normalize(style_feat)
    else:  # AdaIN
        c_proj, _, _            = adain_normalize(content_feat)
        s_proj, s_kernel, s_mean = adain_normalize(style_feat)

    # ─ rearrange (patch swap) ─
    swapped = nearest_patch_swap(c_proj, s_proj, patch_size)

    # ─ interpolate ─
    swapped = alpha * swapped + (1.0 - alpha) * c_proj

    # ─ reconstruct / re-colour ─
    if coding == 'ZCA':
        return zca_colorize(swapped, s_kernel, s_mean)
    else:
        return adain_colorize(swapped, s_kernel, s_mean)


# ── AdaIN helper for decoder fusion ───────────────────────────────────────

def adaptive_instance_norm(content, style):
    """Adaptive Instance Normalisation (Huang & Belongie, 2017)."""
    c_mean = content.mean(dim=[2, 3], keepdim=True)
    c_std  = content.std(dim=[2, 3], keepdim=True) + 1e-7
    s_mean = style.mean(dim=[2, 3], keepdim=True)
    s_std  = style.std(dim=[2, 3], keepdim=True) + 1e-7
    return s_std * (content - c_mean) / c_std + s_mean


# ── Decoder with multi-scale AdaIN fusion ─────────────────────────────────

class AvatarNetDecoder(nn.Module):
    """VGG-mirror decoder from relu4_1 → RGB, with AdaIN fusion at relu3_1.

    Architecture mirrors the VGG-19 decoder from the official TF code:
      conv4_1(512→256) → up → conv3_4..conv3_1 → up → conv2_2..conv2_1
      → up → conv1_2..conv1_1 → output(7×7)

    AdaIN fusion with the encoder-level style features is applied just
    before the conv3_1 layer (matching the 'conv3/conv3_1' entry in the
    original TF decoder table).
    """

    def __init__(self):
        super().__init__()
        # relu4_1 scale
        self.conv4_1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(512, 256, 3), nn.ReLU(inplace=True))
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        # relu3 scale
        self.conv3_4 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True))
        self.conv3_3 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True))
        # ↕ AdaIN fusion with style relu3_1 happens HERE
        self.conv3_1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(256, 128, 3), nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        # relu2 scale
        self.conv2_2 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(128, 64, 3), nn.ReLU(inplace=True))
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        # relu1 scale
        self.conv1_2 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True))
        self.conv1_1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True))

        # output: 7×7 conv, no activation (matching TF combined_decoder)
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7))

    def forward(self, x, style_features=None):
        """
        Args:
            x: decorated feature (1, 512, H, W).
            style_features: dict, e.g. {'relu3_1': tensor (1,256,H',W')}.
        """
        x = self.conv4_1(x)
        x = self.up3(x)
        x = self.conv3_4(x)
        x = self.conv3_3(x)
        x = self.conv3_2(x)

        # Multi-scale fusion at relu3_1
        if style_features is not None and 'relu3_1' in style_features:
            x = adaptive_instance_norm(x, style_features['relu3_1'])

        x = self.conv3_1(x)
        x = self.up2(x)
        x = self.conv2_2(x)
        x = self.conv2_1(x)
        x = self.up1(x)
        x = self.conv1_2(x)
        x = self.conv1_1(x)
        x = self.output(x)
        return x


# ── Feature extraction indices (end-exclusive) in the VGG sequential ──────

_ENC_IDX = {'relu1_1': 4, 'relu2_1': 11, 'relu3_1': 18, 'relu4_1': 31}


# ── Full Avatar-Net model ─────────────────────────────────────────────────

class AvatarNetModel(nn.Module):
    """Avatar-Net: VGG encoder → style decorator → multi-scale decoder."""

    def __init__(self, encoder, decoder,
                 feature_layers=('relu3_1', 'relu4_1'),
                 style_coding='ZCA', patch_size=5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.feature_layers = list(feature_layers)
        self.starting_layer = self.feature_layers[-1]  # deepest
        self.style_coding = style_coding
        self.patch_size = patch_size

    def _extract(self, x, layers):
        """Run encoder once, capture features at requested layers."""
        features = {}
        needed = set(layers)
        max_idx = max(_ENC_IDX[l] for l in needed)
        for i in range(max_idx):
            x = self.encoder[i](x)
            for name, end in _ENC_IDX.items():
                if end == i + 1 and name in needed:
                    features[name] = x
        return features

    @torch.no_grad()
    def forward(self, content, style, alpha=1.0):
        # Encode
        c_feats = self._extract(content, [self.starting_layer])
        s_feats = self._extract(style, self.feature_layers)

        # Style decorator at deepest level
        decorated = style_decorator(
            c_feats[self.starting_layer],
            s_feats[self.starting_layer],
            coding=self.style_coding,
            patch_size=self.patch_size,
            alpha=alpha,
        )

        # Decode with multi-scale fusion
        out = self.decoder(decorated, s_feats)
        return out.clamp(0, 1)
