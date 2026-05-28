"""
MAST (Multi-Adaptation Network) for Arbitrary Style Transfer

Paper: "Arbitrary Style Transfer via Multi-Adaptation Network"
       ACM Multimedia (ACM MM) 2020
       
Authors: Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, 
         Xingjia Pan, Lei Wang, Changsheng Xu

Official Repository: https://github.com/diyiiyiii/Arbitrary-Style-Transfer-via-Multi-Adaptation-Network

Description:
-----------
MAST is a neural style transfer method that uses three complementary attention mechanisms
to achieve high-quality arbitrary style transfer:

1. Content Self-Attention (Content_SA): Position-wise self-attention on content features
   to capture spatial relationships and preserve content structure.

2. Style Self-Attention (Style_SA): Channel-wise self-attention on style features
   to model correlations between different style patterns.

3. Cross-Attention (CA): Attention between content and style features to effectively
   transfer style patterns while maintaining content structure.

The multi-adaptation module operates on relu4_1 features from a pretrained VGG-19 encoder,
combining all three attention mechanisms to produce well-adapted features that are then
decoded to generate the stylized output.

Architecture:
------------
- Encoder: VGG-19 (frozen, pretrained on ImageNet)
- Multi-Adaptation Module: Content_SA + Style_SA + CA
- Decoder: Symmetric architecture to encoder (512→256→128→64→3)

Training:
--------
- Iteration-based training (160,000 iterations)
- Learning rate: 1e-4 with decay schedule: lr = base_lr / (1.0 + 5e-5 * iteration)
- Batch size: 8
- Loss components:
  * Content loss (weight: 1.0): MSE on relu4_1 and relu5_1 features
  * Style loss (weight: 5.0): Mean/std matching on all encoder layers
  * Identity losses: Image-level (weight: 50) and feature-level (weight: 1.0)
  * Discrimination losses: Conditional losses with warmup (weights: 1.0 each, activated when loss_c+loss_s < 10.0)

Required Pretrained Model:
-------------------------
VGG-19 normalized weights (vgg_normalised.pth)
Download: https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view

Usage Example:
-------------
```python
from experiments.reference_methods.style_transfer.training_required.mast import MASTMethod

# Initialize method
config = MASTMethod.get_default_config()
config['batch_size'] = 8  # Adjust as needed
method = MASTMethod(config, pretrained_vgg_path='path/to/vgg_normalised.pth')

# Train
method.train(content_dataset, style_dataset, accelerator)

# Inference
stylized = method.stylize(content_image, style_image, alpha=1.0)
```

Citation:
--------
```bibtex
@inproceedings{deng2020mast,
  title={Arbitrary Style Transfer via Multi-Adaptation Network},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Ma, Chongyang and 
          Pan, Xingjia and Wang, Lei and Xu, Changsheng},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2719--2727},
  year={2020}
}
```

Notes:
-----
- This implementation follows the official MAST codebase closely while integrating
  with our project's infrastructure (checkpointing, data loading, etc.)
- Training is iteration-based (NOT epoch-based) matching the original implementation
- All hyperparameters are taken from the official implementation
- Uses project checkpointing utilities (save_latest_checkpoint, save_model)
  instead of direct accelerator.save_state calls
"""

from .method import Method as MASTMethod
from .net import MASTNet, Multi_Adaptation_Module

__all__ = [
    'MASTMethod',
    'MASTNet',
    'Multi_Adaptation_Module',
]

__version__ = '1.0.0'
__author__ = 'xAILab Bamberg (based on official MAST implementation)'
