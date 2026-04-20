"""
StyleFormer (Real-Time Arbitrary Style Transfer via Parametric Style Composition) implementation.

xAILab Bamberg, University of Bamberg
Based on: https://github.com/Wxl-stars/PytorchStyleFormer

Required Model Downloads:
------------------------

StyleFormer requires the following pre-trained models:

1. VGG-16 Weights (vgg16-397923af.pth):
   Download from: https://download.pytorch.org/models/vgg16-397923af.pth
   
   Local:  /data/local/colorist/checkpoints/reference_methods/styleformer/vgg16-397923af.pth
   Cluster: $WORK/colorist/checkpoints/reference_methods/styleformer/vgg16-397923af.pth

Optional (for pre-trained StyleFormer):
2. Pre-trained StyleFormer Model (gen_*.pt):
   Download from official repository:
   - Google Drive: https://drive.google.com/drive/folders/1l53CJxbMiaU7c17laAT9d8Q_a4arxI28
   - BaiduNetdisk: https://pan.baidu.com/s/1gGHYyIwrtoRxZWQLNWHD1w (Code: kc44)
   
   Local:  /data/local/colorist/checkpoints/reference_methods/styleformer/gen_*.pt
   Cluster: $WORK/colorist/checkpoints/reference_methods/styleformer/gen_*.pt

Training from Scratch:
---------------------

To train StyleFormer from scratch, only the VGG-16 weights are required.
The model will be trained for 100 epochs on your chosen datasets.

Usage:
------
python experiments/reference_methods/pretrain.py \\
    --method styleformer \\
    --dataset_source epistr \\
    --dataset_reference epistr \\
    --data_path /data/local/colorist/data \\
    --checkpoint_base_path /data/local/colorist/checkpoints/reference_methods \\
    --output_path /data/local/colorist/models/pretraining
"""

from .method import Method as StyleFormerMethod

__all__ = ['StyleFormerMethod']
