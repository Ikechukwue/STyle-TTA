"""
AdaIN (Adaptive Instance Normalization) method implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/xunhuang1995/AdaIN-style and https://github.com/naoto0804/pytorch-AdaIN/tree/master


Required Model Downloads:
------------------------

AdaIN requires the following pre-trained models:

1. VGG Encoder (vgg_normalised.pth):
   Download from: https://github.com/naoto0804/pytorch-AdaIN/releases/download/v1.0/vgg_normalised.pth
   
   Local:  /data/local/colorist/checkpoints/reference_methods/adain/vgg_normalised.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/adain/vgg_normalised.pth

Optional (for pre-trained decoder):
2. Pre-trained Decoder (decoder.pth):
   Download from: https://github.com/naoto0804/pytorch-AdaIN/releases/download/v1.0/decoder.pth
   
   Local:  /data/local/colorist/checkpoints/reference_methods/adain/decoder.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/adain/decoder.pth

   
Training from Scratch:
---------------------

To train AdaIN from scratch, only the VGG encoder is required.
"""

from .method import Method as AdaINMethod

__all__ = ['AdaINMethod']