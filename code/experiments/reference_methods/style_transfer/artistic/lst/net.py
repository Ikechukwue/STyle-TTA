"""
LinearStyleTransfer – Learning Linear Transformations for Fast Style Transfer
(Li et al., CVPR 2019).

Architecture adapted from:  https://github.com/sunshineatnoon/LinearStyleTransfer

The method learns a *linear transformation matrix* to map content features
into the style space.  A small CNN predicts Gram-like matrices for content and
style, which are multiplied and applied in a compressed feature space (default
32-d).  Supports two feature levels: relu3_1 (256-ch) and relu4_1 (512-ch).
"""

import torch
import torch.nn as nn


# ── Small CNN for Gram-like matrix prediction ──────────────────────────────

class CNN(nn.Module):
    """Predict a (matrix_size × matrix_size) transformation matrix from a
    feature map via convolutions + Gram + FC."""

    def __init__(self, in_channels, matrix_size=32):
        super().__init__()
        if in_channels == 256:  # relu3_1
            self.convs = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, matrix_size, 3, 1, 1),
            )
        else:  # relu4_1 (512)
            self.convs = nn.Sequential(
                nn.Conv2d(512, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, matrix_size, 3, 1, 1),
            )
        self.fc = nn.Linear(matrix_size * matrix_size,
                            matrix_size * matrix_size)

    def forward(self, x):
        out = self.convs(x)
        b, c, h, w = out.size()
        out = out.view(b, c, -1)
        out = torch.bmm(out, out.transpose(1, 2)).div(h * w)
        return self.fc(out.view(b, -1))


# ── Linear transformation module ──────────────────────────────────────────

class MulLayer(nn.Module):
    """Learns a linear transformation between content and style feature
    spaces in a compressed channel space of size *matrix_size*."""

    def __init__(self, in_channels, matrix_size=32):
        super().__init__()
        self.snet = CNN(in_channels, matrix_size)
        self.cnet = CNN(in_channels, matrix_size)
        self.matrix_size = matrix_size
        self.compress = nn.Conv2d(in_channels, matrix_size, 1, 1, 0)
        self.unzip = nn.Conv2d(matrix_size, in_channels, 1, 1, 0)

    def forward(self, cF, sF):
        """Apply learned linear style transfer.

        Args:
            cF: content features (B, C, H, W).
            sF: style features  (B, C, H, W).
        Returns:
            Stylised features  (B, C, H, W).
        """
        # mean-centre
        cb, cc, ch, cw = cF.size()
        cMean = cF.view(cb, cc, -1).mean(dim=2, keepdim=True).unsqueeze(3)
        cF_c = cF - cMean.expand_as(cF)

        sb, sc, sh, sw = sF.size()
        sMean = sF.view(sb, sc, -1).mean(dim=2, keepdim=True).unsqueeze(3)
        sF_c = sF - sMean.expand_as(sF)

        # compress → transform → decompress
        compressed = self.compress(cF_c)
        b, c, h, w = compressed.size()
        compressed = compressed.view(b, c, -1)

        cMatrix = self.cnet(cF_c).view(b, self.matrix_size, self.matrix_size)
        sMatrix = self.snet(sF_c).view(b, self.matrix_size, self.matrix_size)
        trans = torch.bmm(sMatrix, cMatrix)
        transfeature = torch.bmm(trans, compressed).view(b, c, h, w)

        out = self.unzip(transfeature) + sMean.expand_as(cF)
        return out


# ── Full network wrapper ──────────────────────────────────────────────────

class LSTNet(nn.Module):
    """Encoder → MulLayer → Decoder pipeline."""

    def __init__(self, encoder, decoder, matrix):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.matrix = matrix

    @torch.no_grad()
    def forward(self, content, style):
        cF = self.encoder(content)
        sF = self.encoder(style)
        feature = self.matrix(cF, sF)
        return self.decoder(feature).clamp(0, 1)
