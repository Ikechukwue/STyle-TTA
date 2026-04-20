"""
EFDM Network Architecture (inference only)

xAILab Bamberg, University of Bamberg
Based on: https://github.com/YBZh/EFDM/tree/main/ArbitraryStyleTransfer

Kept for inference:
- EFDMNet (VGG encoder + EFDM matching + decoder)
- exact_feature_distribution_matching (core EFDM algorithm)

Removed (training only):
- calc_content_loss, calc_style_loss, training forward()
"""

import torch
import torch.nn as nn


def exact_feature_distribution_matching(content_feat, style_feat):
    """
    EFDM: Exact Feature Distribution Matching via Sort-Matching.

    Matches the exact empirical distribution of features by sorting and matching CDFs.
    Implicitly matches all moments efficiently without assuming Gaussian distributions.

    Reference:
        Zhang et al. "Exact Feature Distribution Matching for Arbitrary Style Transfer
        and Domain Generalization." CVPR 2022.
    """
    assert content_feat.size() == style_feat.size()
    B, C, W, H = content_feat.size()
    value_content, index_content = torch.sort(content_feat.view(B, C, -1))
    value_style, _ = torch.sort(style_feat.view(B, C, -1))
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(B, C, -1) + (
        value_style.gather(-1, inverse_index) - content_feat.view(B, C, -1).detach()
    )
    return new_content.view(B, C, W, H)


class EFDMNet(nn.Module):
    """
    EFDM network: VGG encoder (frozen) + EFDM matching + decoder.

    Args:
        encoder: VGG encoder (first 31 layers up to relu4_1)
        decoder: Decoder network
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])    # -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])   # -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # -> relu4_1
        self.decoder = decoder

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode(self, x):
        for i in range(4):
            x = getattr(self, f'enc_{i+1}')(x)
        return x
