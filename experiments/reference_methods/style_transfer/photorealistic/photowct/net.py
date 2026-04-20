"""
PhotoWCT – A Closed-form Solution to Photorealistic Image Stylization.

Network architecture adapted from the official NVIDIA repo:
    https://github.com/NVIDIA/FastPhotoStyle

The method uses 4 levels of VGG-19 (normalised) encoder slices paired with
4 reconstruction decoders that use MaxUnpool for spatial correspondence.
At inference time a coarse-to-fine cascade is run:
    encode(content, L4) → WCT → decode4 → encode(result, L3) → WCT → ... → out.

The WCT (Whitening and Coloring Transform) aligns the content feature
statistics to the style by ZCA-whitening the content features and then
re-coloring them with the style covariance/mean.

After stylization, a photorealistic smoothing step is applied (either
Laplacian-based propagation or Guided Image Filtering) to ensure local
consistency with the content image structure.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# VGG-19 Encoder (normalised), truncated at different levels
# Level 1: up to relu1_1  (64 ch)
# Level 2: up to relu2_1  (128 ch)
# Level 3: up to relu3_1  (256 ch)
# Level 4: up to relu4_1  (512 ch)
# ---------------------------------------------------------------------------


class VGGEncoder(nn.Module):
    """VGG-19 encoder (normalised weights) truncated at a given level.

    Uses MaxPool2d with ``return_indices=True`` so that the corresponding
    decoder can use :class:`nn.MaxUnpool2d` for up-sampling.

    ``forward`` returns the feature map plus pooling indices/sizes for all
    intermediate pooling operations.

    ``forward_multiple`` returns the feature map at *every* encoder level
    so that a multi-level cascade can extract style features in one pass.
    """

    def __init__(self, level: int):
        super().__init__()
        self.level = level

        # 224 × 224
        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)

        self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu1_1 = nn.ReLU(inplace=True)

        if level < 2:
            return

        self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2_1 = nn.ReLU(inplace=True)

        if level < 3:
            return

        self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu3_1 = nn.ReLU(inplace=True)

        if level < 4:
            return

        self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_3 = nn.ReLU(inplace=True)

        self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu3_4 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu4_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv0(x)

        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        if self.level < 2:
            return out

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_idx = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        if self.level < 3:
            return out, pool1_idx, pool1.size()

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_idx = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        if self.level < 4:
            return out, pool1_idx, pool1.size(), pool2_idx, pool2.size()

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, pool1_idx, pool1.size(), pool2_idx, pool2.size(), pool3_idx, pool3.size()

    def forward_multiple(self, x):
        """Extract features at *all* encoder levels in a single forward pass.

        Used for the style image to get features for the 4-level cascade.

        Returns:
            (feat4, feat3, feat2, feat1)  – features from deepest to shallowest.
        """
        out = self.conv0(x)

        out = self.pad1_1(out)
        out = self.conv1_1(out)
        out = self.relu1_1(out)

        if self.level < 2:
            return out

        out1 = out

        out = self.pad1_2(out)
        out = self.conv1_2(out)
        pool1 = self.relu1_2(out)

        out, pool1_idx = self.maxpool1(pool1)

        out = self.pad2_1(out)
        out = self.conv2_1(out)
        out = self.relu2_1(out)

        if self.level < 3:
            return out, out1

        out2 = out

        out = self.pad2_2(out)
        out = self.conv2_2(out)
        pool2 = self.relu2_2(out)

        out, pool2_idx = self.maxpool2(pool2)

        out = self.pad3_1(out)
        out = self.conv3_1(out)
        out = self.relu3_1(out)

        if self.level < 4:
            return out, out2, out1

        out3 = out

        out = self.pad3_2(out)
        out = self.conv3_2(out)
        out = self.relu3_2(out)

        out = self.pad3_3(out)
        out = self.conv3_3(out)
        out = self.relu3_3(out)

        out = self.pad3_4(out)
        out = self.conv3_4(out)
        pool3 = self.relu3_4(out)
        out, pool3_idx = self.maxpool3(pool3)

        out = self.pad4_1(out)
        out = self.conv4_1(out)
        out = self.relu4_1(out)

        return out, out3, out2, out1


# ---------------------------------------------------------------------------
# VGG-19 Decoder (mirrors encoder with MaxUnpool)
# ---------------------------------------------------------------------------


class VGGDecoder(nn.Module):
    """Reconstruction decoder for a given VGG encoder level.

    Uses :class:`nn.MaxUnpool2d` with pooling indices from the encoder to
    up-sample spatially, preserving structure important for photorealism.
    """

    def __init__(self, level: int):
        super().__init__()
        self.level = level

        if level > 3:
            self.pad4_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)
            self.relu4_1 = nn.ReLU(inplace=True)

            self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            self.pad3_4 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_4 = nn.ReLU(inplace=True)

            self.pad3_3 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_3 = nn.ReLU(inplace=True)

            self.pad3_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
            self.relu3_2 = nn.ReLU(inplace=True)

        if level > 2:
            self.pad3_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)
            self.relu3_1 = nn.ReLU(inplace=True)

            self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            self.pad2_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
            self.relu2_2 = nn.ReLU(inplace=True)

        if level > 1:
            self.pad2_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)
            self.relu2_1 = nn.ReLU(inplace=True)

            self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

            self.pad1_2 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
            self.relu1_2 = nn.ReLU(inplace=True)

        if level > 0:
            self.pad1_1 = nn.ReflectionPad2d((1, 1, 1, 1))
            self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, pool1_idx=None, pool1_size=None,
                pool2_idx=None, pool2_size=None,
                pool3_idx=None, pool3_size=None):
        out = x

        if self.level > 3:
            out = self.pad4_1(out)
            out = self.conv4_1(out)
            out = self.relu4_1(out)
            out = self.unpool3(out, pool3_idx, output_size=pool3_size)

            out = self.pad3_4(out)
            out = self.conv3_4(out)
            out = self.relu3_4(out)

            out = self.pad3_3(out)
            out = self.conv3_3(out)
            out = self.relu3_3(out)

            out = self.pad3_2(out)
            out = self.conv3_2(out)
            out = self.relu3_2(out)

        if self.level > 2:
            out = self.pad3_1(out)
            out = self.conv3_1(out)
            out = self.relu3_1(out)
            out = self.unpool2(out, pool2_idx, output_size=pool2_size)

            out = self.pad2_2(out)
            out = self.conv2_2(out)
            out = self.relu2_2(out)

        if self.level > 1:
            out = self.pad2_1(out)
            out = self.conv2_1(out)
            out = self.relu2_1(out)
            out = self.unpool1(out, pool1_idx, output_size=pool1_size)

            out = self.pad1_2(out)
            out = self.conv1_2(out)
            out = self.relu1_2(out)

        if self.level > 0:
            out = self.pad1_1(out)
            out = self.conv1_1(out)

        return out


# ---------------------------------------------------------------------------
# PhotoWCT: full network wrapping 4 encoder-decoder pairs + WCT
# ---------------------------------------------------------------------------


class PhotoWCTNet(nn.Module):
    """PhotoWCT – Photorealistic Image Stylization via WCT.

    Uses 4 VGG encoder/decoder pairs.  The deepest level (level 4, relu4_1)
    encoder also provides multi-level style features via ``forward_multiple``.
    The forward pass cascades from level 4 → 1, applying the Whitening and
    Coloring Transform (WCT) at each level.

    Decoders use MaxUnpool with indices from the content encoder to
    preserve spatial structure important for photorealism.
    """

    def __init__(self):
        super().__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)

    # ── WCT core ────────────────────────────────────────────────────────

    @staticmethod
    def wct_core(cont_feat, styl_feat):
        """ZCA-whiten *cont_feat* and re-colour with *styl_feat* statistics.

        Both inputs are 2-D: (C, N) where N = H*W.
        """
        cFSize = cont_feat.size()
        c_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
        cont_feat = cont_feat - c_mean

        iden = torch.eye(cFSize[0], device=cont_feat.device, dtype=cont_feat.dtype)
        contentConv = torch.mm(cont_feat, cont_feat.t()).div(cFSize[1] - 1) + iden
        c_u, c_e, c_v = torch.svd(contentConv, some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0] - 1, -1, -1):
            if c_e[i] >= 1e-5:
                k_c = i + 1
                break

        sFSize = styl_feat.size()
        s_mean = torch.mean(styl_feat, 1)
        styl_feat = styl_feat - s_mean.unsqueeze(1).expand_as(styl_feat)
        styleConv = torch.mm(styl_feat, styl_feat.t()).div(sFSize[1] - 1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0] - 1, -1, -1):
            if s_e[i] >= 1e-5:
                k_s = i + 1
                break

        c_d = (c_e[:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, :k_c], torch.diag(c_d))
        step2 = torch.mm(step1, c_v[:, :k_c].t())
        whiten_cF = torch.mm(step2, cont_feat)

        s_d = (s_e[:k_s]).pow(0.5)
        targetFeature = torch.mm(
            torch.mm(torch.mm(s_v[:, :k_s], torch.diag(s_d)), s_v[:, :k_s].t()),
            whiten_cF,
        )
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def feature_wct(self, cF, sF, alpha=1.0):
        """Apply WCT with alpha blending.  cF and sF are (C, H, W)."""
        C, H, W = cF.size()
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)
        target = self.wct_core(cFView, sFView).view_as(cF)
        blended = alpha * target + (1.0 - alpha) * cF
        return blended.float().unsqueeze(0)

    # ── inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, content, style, alpha=1.0):
        """Coarse-to-fine multi-level photorealistic stylisation.

        Args:
            content: (1, 3, H, W) in [0, 1].
            style:   (1, 3, H, W) in [0, 1].
            alpha:   style weight (0 = content, 1 = full transfer).
        Returns:
            Stylised image (1, 3, H, W) clamped to [0, 1].
        """
        # Extract style features at all 4 levels in one pass
        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(style)

        # Level 4
        cF4, cp1_idx, cp1, cp2_idx, cp2, cp3_idx, cp3 = self.e4(content)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)
        csF4 = self.feature_wct(cF4, sF4, alpha)
        Im4 = self.d4(csF4, cp1_idx, cp1, cp2_idx, cp2, cp3_idx, cp3)

        # Level 3
        cF3, cp1_idx, cp1, cp2_idx, cp2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.feature_wct(cF3, sF3, alpha)
        Im3 = self.d3(csF3, cp1_idx, cp1, cp2_idx, cp2)

        # Level 2
        cF2, cp1_idx, cp1 = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.feature_wct(cF2, sF2, alpha)
        Im2 = self.d2(csF2, cp1_idx, cp1)

        # Level 1
        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.feature_wct(cF1, sF1, alpha)
        Im1 = self.d1(csF1)

        return Im1.clamp(0, 1)
