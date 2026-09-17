"""
CAST (Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning) implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/zyxElsa/CAST_pytorch

Paper: "Domain Enhanced Arbitrary Image Style Transfer via Contrastive Learning"
Authors: Yuxin Zhang, Fan Tang, Weiming Dong, Haibin Huang, Chongyang Ma, Tong-Yee Lee, Changsheng Xu
Conference: SIGGRAPH 2022

Required Model Downloads:
------------------------

CAST requires the following pre-trained models:

1. VGG Encoder (vgg_normalised.pth):
   Download from: https://drive.google.com/file/d/1DKYRWJUKbmrvEba56tuihy1N6VrNZFwl/view?usp=sharing
   
   Local:  /data/local/colorist/checkpoints/reference_methods/cast/vgg_normalised.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/cast/vgg_normalised.pth

2. Style VGG (style_vgg.pth) for style classification:
   Download from: https://drive.google.com/file/d/12JKlL6QsVWkz6Dag54K59PAZigFBS6PQ/view?usp=sharing
   
   Local:  /data/local/colorist/checkpoints/reference_methods/cast/style_vgg.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/cast/style_vgg.pth

Optional (for pre-trained model):
3. Pre-trained CAST model (CAST_model/*.pth):
   Download from: https://drive.google.com/file/d/11dZqu95QfnAgkzgR1NTJfQutz8JlwRY8/view?usp=sharing
   
   Local:  /data/local/colorist/checkpoints/reference_methods/cast/cast_pretrained.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/cast/cast_pretrained.pth

   
Training from Scratch:
---------------------

To train CAST from scratch, both VGG encoder and style VGG are required.
The model uses contrastive learning with domain-specific style representations.

Key Features:
- Adaptive Instance Normalization (AdaIN) for style transfer
- Contrastive learning with InfoNCE loss
- Style discriminator with queue-based memory bank
- Cycle consistency loss for better content preservation
- Dual decoders for bidirectional style transfer
"""

from .method import Method as CASTMethod

__all__ = ['CASTMethod']
