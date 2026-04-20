"""
StyTr^2 (Image Style Transfer with Transformers)

Official Paper: "StyTr^2: Image Style Transfer with Transformers"
Official Code: https://github.com/diyiiyiii/StyTR-2

IMPORTANT: Before using this method, you need to download pretrained VGG weights.

Required Download:
------------------
1. VGG Normalised Weights (REQUIRED for training and inference):
   - Download: https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M
   - File: vgg_normalised.pth
   - Save to: Your centralized pretrained models directory

Optional Pretrained Models (for inference without training):
------------------------------------------------------------
2. Pretrained Transformer Module:
   - Download: https://drive.google.com/file/d/1C3xzTOWx8dUXOFMPHfJYY-HJ7xEJRo3E
   - File: transformer_iter_160000.pth

3. Pretrained Decoder:
   - Download: https://drive.google.com/file/d/1C3xzTOWx8dUXOFMPHfJYY-HJ7xEJRo3E
   - File: decoder_iter_160000.pth

4. ViT Embeddings (optional, for alternative initialization):
   - Download: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

Usage:
------
```python
from experiments.reference_methods.style_transfer.training_required.stytr2.method import StyTr2Method

# For inference with pretrained weights
method = StyTr2Method(pretrained_weights='/path/to/checkpoint.pth')
stylized_image = method(content_image, style_image)

# For training
method = StyTr2Method()
method.initialize_for_training(pretrained_vgg_path='/path/to/vgg_normalised.pth')
method.train(
    dataset_source='epistr',
    dataset_reference='epistr',
    checkpoint_path='/path/to/checkpoints/',
    final_model_path='/path/to/final_model.pth',
    accelerator=accelerator
)
```

Architecture:
-------------
- VGG-19 encoder (5 levels: relu1_1 to relu5_1, 44 layers total)
- PatchEmbed: Conv2d(3, 512, kernel_size=8, stride=8) for 8x8 patches
- Transformer:
  * d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1
  * Content Encoder: 3 TransformerEncoderLayers
  * Style Encoder: 3 TransformerEncoderLayers (separate from content)
  * Decoder: 3 TransformerDecoderLayers
- Content-Aware Positional Embedding (CAPE): AdaptiveAvgPool2d(18) + Conv2d(512,512,1,1)
- Progressive Decoder: Upsampling 512→256→128→64→3 channels

Training Configuration:
----------------------
- Learning rate: 5e-4 (warmup) → 2e-4 (decay after 10k iters)
- LR warmup: lr = 5e-4 * 0.1 * (1.0 + 3e-4 * iteration) for first 10k iters
- LR decay: lr = 2e-4 / (1.0 + 1e-5 * (iteration - 10000)) after warmup
- Batch size: 8
- Max iterations: 160,000
- Optimizer: Adam (default betas)
- Image preprocessing: Resize(512) → RandomCrop(256)
- Native size: 256×256

Loss Components:
---------------
- Content loss (weight=7.0): Applied to relu4_1 and relu5_1 features
- Style loss (weight=10.0): Applied to all 5 VGG levels (mean/std matching)
- Identity loss 1 (weight=70): Pixel-level reconstruction (content→content)
- Identity loss 2 (weight=1): Feature-level reconstruction (style→style)

Citation:
---------
@article{deng2022stytr2,
  title={StyTr$^2$: Image Style Transfer with Transformers},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Ma, Chongyang and Pan, Xingjia and Wang, Lei and Xu, Changsheng},
  journal={arXiv preprint arXiv:2105.14576},
  year={2022}
}
"""

from .method import Method as StyTr2Method
from .net import StyTrans, VGGEncoder, PatchEmbed, Transformer, Decoder

__all__ = ['StyTr2Method', 'StyTrans', 'VGGEncoder', 'PatchEmbed', 'Transformer', 'Decoder']
