"""
AdaAttN (Adaptive Attentional Instance Normalization) implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/Huage001/AdaAttN

Paper: "AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer"
Authors: Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling Wang, Xin Li, 
         Zhengxing Sun, Qian Li, Errui Ding
Conference: ICCV 2021

Required Model Downloads:
------------------------

1. VGG Encoder Weights (vgg_normalised.pth)
   - Download from: https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing
   - Size: ~77MB
   
   Local path:
   /data/local/colorist/checkpoints/reference_methods/adaattn/vgg_normalised.pth
   
   Cluster path:
   $WORK/colorist/checkpoints/reference_methods/adaattn/vgg_normalised.pth

Method Overview:
---------------

AdaAttN improves upon AdaIN by incorporating attention mechanisms that allow
for more semantically meaningful style transfer. Instead of uniformly applying
style statistics across all spatial locations, AdaAttN computes attention weights
between content and style features, enabling region-specific style transfer.

Key features:
- Multi-scale attention at relu3_1, relu4_1, and relu5_1
- Attention-weighted adaptive instance normalization
- Optional skip connection at relu3 for better detail preservation
- Shallow layer feature concatenation for richer semantic matching

Architecture:
- VGG-19 encoder (frozen, pre-trained on ImageNet)
- Transformer module with AdaAttN at multiple scales
- Decoder that mirrors encoder structure
- Optional AdaAttN_3 module for skip connection

Training Configuration:
- Content dataset: COCO train2014
- Style dataset: WikiArt
- Batch size: 8
- Learning rate: 0.0002 (Adam, beta1=0.5)
- Epochs: 100 constant + 100 exponential decay (lr *= 0.3^epoch)
- Loss weights: content=0, global=10, local=3
- Loss computed at: relu1_1 through relu4_1
- Image size: 512 load -> 256 crop
"""

from .method import Method as AdaAttNMethod

__all__ = ['AdaAttNMethod']
