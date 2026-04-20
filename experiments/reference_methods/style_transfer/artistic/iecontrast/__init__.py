"""
xAILab Bamberg
University of Bamberg

@description:
IEContrAST (Internal-external Contrastive Artistic Style Transfer) method.
Based on official PyTorch implementation: https://github.com/HalbertCH/IEContraAST

Paper: "Artistic Style Transfer with Internal-external Learning and Contrastive Learning"
Conference: NeurIPS 2021

To use this method, you need to download pre-trained VGG weights:
- VGG-19 weights: https://drive.google.com/file/d/1BinnwM5AmKMNsg0m8Pjpsd3ZvHbSr5by/view?usp=sharing
- Save to: experiments/reference_methods/pretrained_models/vgg_normalised.pth

The method can then be initialized for training with these VGG weights.
"""

from .method import Method as IEContrASTMethod

__all__ = ['IEContrASTMethod']
