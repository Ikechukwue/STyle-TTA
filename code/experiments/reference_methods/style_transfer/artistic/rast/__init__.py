"""
RAST (Restorable Arbitrary Style Transfer) method implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/YingnanMa/RAST

Paper: "RAST: Restorable Arbitrary Style Transfer via Multi-restoration"
Authors: Yingnan Ma, Xudong Li et al.
Conference: WACV 2023


Required Model Downloads:
------------------------

RAST requires the following pre-trained models:

1. VGG Encoder (vgg_normalised.pth):
   Google Drive:  https://drive.google.com/file/d/1cI6ubAziMdOsSJZEvfofW-iCtnCmsONL/view
   (Same normalised VGG-19 used by AdaIN, SANet, etc.)

   Local:   /data/local/colorist/checkpoints/reference_methods/rast/vgg_normalised.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/rast/vgg_normalised.pth

2. Decoder (decoder_iter_160000.pth):
   GitHub: https://github.com/YingnanMa/RAST/raw/main/model/decoder_iter_160000.pth

   Local:   .../reference_methods/rast/decoder_iter_160000.pth
   Cluster: $WORK/.../rast/decoder_iter_160000.pth

3. Transformer (transformer_iter_160000.pth):
   GitHub: https://github.com/YingnanMa/RAST/raw/main/model/transformer_iter_160000.pth

   Local:   .../reference_methods/rast/transformer_iter_160000.pth
   Cluster: $WORK/.../rast/transformer_iter_160000.pth


Merge Weights:
--------------

See the docstring in method.py for the full merge script.
In short:
    >>> import torch
    >>> vgg = torch.load('vgg_normalised.pth', weights_only=True)
    >>> dec = torch.load('decoder_iter_160000.pth', weights_only=True)
    >>> from code.experiments.reference_methods.color_transfer\
    ...     .artistic.rast.net import Transform
    >>> transform = Transform(in_planes=512)
    >>> trans = torch.load('transformer_iter_160000.pth', weights_only=True)
    >>> transform.load_state_dict(trans)
    >>> torch.save({
    ...     'encoder': vgg,
    ...     'decoder': dec,
    ...     'transformer': transform.state_dict(),
    ... }, 'rast_merged.pth')


Related versions:
-----------------

  - RAST 2.0: https://github.com/YingnanMa/RAST-2.0
    (includes training code with IEAST-replaced strategy)
  - RAST 4.0: https://github.com/YingnanMa/RAST-4.0  (ICSM 2025)
    (content leakage correction; same inference architecture)
"""

from .method import Method as RASTMethod

__all__ = ["RASTMethod"]
