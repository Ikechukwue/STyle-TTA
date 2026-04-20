"""
SANET (Style-Attentional Network) method implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/GlebSBrykin/SANET

Paper: "Arbitrary Style Transfer with Style-Attentional Networks"
Authors: Dae Young Park, Kwang Hee Lee
Conference: CVPR 2019


Required Model Downloads:
------------------------

SANET requires the following pre-trained models:

1. VGG Encoder (vgg_normalised.pth):
   Download from: https://yadi.sk/d/7IrysY8q8dtneQ
   (Same as AdaIN VGG encoder)
   
   Local:  /data/local/colorist/checkpoints/reference_methods/sanet/vgg_normalised.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/sanet/vgg_normalised.pth

Optional (for pre-trained decoder and transformer):
2. Pre-trained Decoder (decoder_iter_500000.pth):
   Download from: https://yadi.sk/d/xsZ7j6FhK1dmfQ
   
   Local:  /data/local/colorist/checkpoints/reference_methods/sanet/decoder_iter_500000.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/sanet/decoder_iter_500000.pth

3. Pre-trained Transformer (transformer_iter_500000.pth):
   Download from: https://yadi.sk/d/GhQe3g_iRzLKMQ
   
   Local:  /data/local/colorist/checkpoints/reference_methods/sanet/transformer_iter_500000.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/sanet/transformer_iter_500000.pth

   
Training from Scratch:
---------------------

To train SANET from scratch, only the VGG encoder is required.
The decoder and transformer networks will be initialized randomly and trained.

Training uses:
- MS-COCO 2014 for content images
- WikiArt for style images
- 160,000 iterations
- Batch size of 5
- Learning rate 1e-4 with decay 5e-5
"""

from .method import Method as SANETMethod

__all__ = ['SANETMethod']
