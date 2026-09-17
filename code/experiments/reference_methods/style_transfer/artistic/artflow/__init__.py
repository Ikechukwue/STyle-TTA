"""
ArtFlow (Unbiased Image Style Transfer via Reversible Neural Flows) implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/pkuanjie/ArtFlow

Paper: "ArtFlow: Unbiased Image Style Transfer via Reversible Neural Flows"
Authors: Jie An, Siyu Huang, Yibing Song, Dejing Dou, Wei Liu, Jiebo Luo
Conference: CVPR 2021

Required Model Downloads:
------------------------

ArtFlow requires the following pre-trained models:

1. VGG Encoder (vgg_normalised.pth):
   Download from: https://github.com/naoto0804/pytorch-AdaIN/releases/download/v1.0/vgg_normalised.pth
   OR: https://github.com/xunhuang1995/AdaIN-style
   
   Local:  /data/local/colorist/checkpoints/reference_methods/artflow/vgg_normalised.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/artflow/vgg_normalised.pth

Optional (for pre-trained Glow models):
2. Pre-trained ArtFlow models:
   Download from: https://drive.google.com/drive/folders/1w2fHgSBYwjplfeCXI8eOGYpi69CpJBTE?usp=sharing
   
   Available models:
   - ArtFlow-AdaIN/glow.pth
   - ArtFlow-WCT/glow.pth
   - ArtFlow-AdaIN-Portrait/glow.pth (for portrait style transfer)
   
   Local:  /data/local/colorist/checkpoints/reference_methods/artflow/
   Cluster: $WORK/colorist/checkpoints/reference_methods/artflow/

Training from Scratch:
---------------------

To train ArtFlow from scratch, only the VGG encoder is required.
The method supports three operators:
- adain: Adaptive Instance Normalization (fast, general purpose)
- wct: Whitening and Coloring Transform (higher quality, slower)
- decorator: Patch-based style decorator (best quality, slowest)

Architecture:
------------

ArtFlow uses a Glow-based reversible neural flow architecture to avoid content leakage
in style transfer. The key innovation is performing style transfer in the latent space
(z space) rather than in image space, which prevents the content from leaking into the
stylized result during continuous style transfer.

Hyperparameters (Official):
---------------------------
- learning_rate: 1e-4
- lr_decay: 5e-5
- max_iter: 160000
- batch_size: 4
- n_flow: 8 (flows per block)
- n_block: 2 (number of blocks)
- style_weight: 1.0
- content_weight: 0.1
- mse_weight: 0
- native_size: 256 (512 resize -> 256 crop)
"""

from .method import Method as ArtFlowMethod

__all__ = ['ArtFlowMethod']
