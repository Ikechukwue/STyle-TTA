"""
StyleID (Style Injection in Diffusion) - Training-Free Style Transfer

xAILab Bamberg, University of Bamberg
Based on: https://github.com/jiwoogit/StyleID

TRAINING-FREE METHOD
--------------------
This method uses pretrained Stable Diffusion models and requires NO pretraining
or fine-tuning. Models are automatically downloaded from HuggingFace on first use.

Paper: "Style Injection in Diffusion: A Training-free Approach for Adapting 
        Large-scale Diffusion Models for Style Transfer"
Authors: Jiwoo Chung, Sangeek Hyun, Jae-Pil Heo
Conference: CVPR 2024 (Highlight)
Paper: https://arxiv.org/abs/2312.09008
Project Page: https://jiwoogit.github.io/StyleID_site/

Required Pretrained Models:
--------------------------
Stable Diffusion models are automatically downloaded via the diffusers library:

- SD 1.4: CompVis/stable-diffusion-v1-4 (default)
- SD 1.5: runwayml/stable-diffusion-v1-5
- SD 2.0: stabilityai/stable-diffusion-2 (no longer available)
- SD 2.1: stabilityai/stable-diffusion-2-1 (no longer available)
- SD 2.1-base: stabilityai/stable-diffusion-2-1-base (no longer available)

First-time download: ~5GB per model
Cached location: ~/.cache/huggingface/

GPU Requirements:
----------------
- Minimum VRAM: 20GB (for full inference pipeline)
- Recommended: A100 (40GB) or equivalent
- Supports float16 for memory efficiency

No Manual Downloads Required!
-----------------------------
Unlike training-required methods, you do NOT need to manually download checkpoints.
Simply call method.initialize() and models will be fetched automatically.

Usage:
------
```python
from code.experiments.reference_methods.style_transfer.training_free.styleid.method import StyleIDMethod

# Initialize method with default parameters
method = StyleIDMethod(device='cuda')
method.initialize()  # Downloads SD models on first use

# Inference
content = torch.randn(1, 3, 512, 512)  # [0, 1] range
style = torch.randn(1, 3, 512, 512)
output = method(content, style)

# Check default configuration
config = method.get_default_config()
print(f"gamma: {config['gamma']}, T: {config['T']}, ddim_steps: {config['ddim_steps']}")
```

Parameter Configuration:
-----------------------
Parameters are defined in method.py via get_default_config() and can be
modified by editing that method. Default values (matching official implementation):

- gamma (0-1): Query preservation
  * Default: 0.75 (balanced content/style)
  * 0.3: High style fidelity (recommended for strong style transfer)
  * 0.9: Strong content preservation
  
- T (>1): Attention temperature
  * Default: 1.5
  * 1.0-2.0: Typical range
  * Higher = stronger style transfer
  
- ddim_steps: Sampling steps
  * Default: 50 (official run_styleid.py default)
  * More steps = better quality, slower inference
  * 20-100 typical range

- start_step: Step index to start injection
  * Default: 49 (official default)
  * Only applies style injection for steps where index < start_step
  * Effectively applies injection only in the last 1-2 denoising steps

- injection_layers: UNet decoder layers for style injection
  * Default: [6, 7, 8, 9, 10, 11] (official: '6,7,8,9,10,11')

- sd_version: Stable Diffusion version
  * Default: '1.4' (original implementation uses sd-v1-4)
  * Options: '1.4', '1.5', '2.0', '2.1', '2.1-base'

To modify parameters, edit the get_default_config() method in method.py.

Method Overview:
---------------
StyleID leverages pretrained Stable Diffusion models for high-quality style transfer:

1. VAE Encoding: Encode content and style images to latent space
2. DDIM Inversion: Invert both images to get noise + attention features
3. Style Injection: During sampling, inject style attention (K, V) while preserving content query (Q)
4. Attention Mixing: Q_mixed = gamma * Q_content + (1-gamma) * Q_stylized
5. Temperature Scaling: Scale attention by temperature T
6. VAE Decoding: Decode final latent to image space

Comparison with Training-Required Methods:
------------------------------------------
| Aspect           | Training-Required | Training-Free (StyleID) |
|------------------|-------------------|-------------------------|
| Setup Time       | Hours (training)  | Minutes (download)      |
| Pretraining      | Required          | Not needed              |
| Inference Speed  | Fast (~0.01s)     | Slower (~3-5s)          |
| GPU Memory       | Low (~2GB)        | High (~20GB)            |
| Quality          | Good              | Excellent               |

Known Limitations:
-----------------
- Memory intensive: Requires high-end GPU (20GB+ VRAM)
- Slow inference: ~3-5 seconds per image pair (not suitable for real-time)
- Fixed resolution: Native 512×512 (interpolated for other sizes)
- Not suitable for training augmentation (too slow)

Performance Notes:
-----------------
- Inference time: ~3-5 seconds per image pair on A100
- GPU memory: ~20GB VRAM required
- Quality: State-of-the-art diffusion-based style transfer
- Best use case: High-quality evaluation and comparison

Dependencies:
------------
- diffusers>=0.21.0
- transformers>=4.25.0
- accelerate>=0.20.0

Install with: pip install diffusers transformers accelerate
"""

from .method import Method as StyleIDMethod

__all__ = ['StyleIDMethod']
