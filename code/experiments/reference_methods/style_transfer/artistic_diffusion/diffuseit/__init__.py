"""
DiffuseIT - Diffusion-based Image Translation using Disentangled Style and Content Representation

xAILab Bamberg, University of Bamberg
Based on: https://github.com/cyclomon/DiffuseIT

TRAINING-FREE METHOD
--------------------
This method uses pretrained diffusion models and requires NO pretraining
or fine-tuning. Models must be manually downloaded from Google Drive.

Paper: "Diffusion-based Image Translation using Disentangled Style and Content Representation"
Authors: Gihyun Kwon, Jong Chul Ye
Conference: ICLR 2023
Paper: https://arxiv.org/abs/2209.15264
Code: https://github.com/cyclomon/DiffuseIT

Required Pretrained Models:
--------------------------
Models must be manually downloaded from Google Drive:

1. ImageNet 256×256 Diffusion Model (REQUIRED):
   - Download: https://drive.google.com/file/d/1kfCPMZLaAcpoIcvzTHwVVJ_qDetH-Rns/view?usp=sharing
   - Save to: ~/.cache/colorist/diffuseit/256x256_diffusion_uncond.pt
   - Size: ~5GB
   - Checksum: (see README for verification)

2. FFHQ 256×256 Diffusion Model (Optional - for face images):
   - Download: https://drive.google.com/file/d/1-oY7JjRtET4QP3PIWg3ilxAo4VfjCa3J/view?usp=sharing
   - Save to: ~/.cache/colorist/diffuseit/ffhq_10m.pt
   - Size: ~5GB
   - Only needed when use_ffhq=True

3. ArcFace Identity Model (Optional - only with FFHQ):
   - Download: https://drive.google.com/file/d/1SJa5qVNM6jGZdmsnUsGNhjtrssGYuJfT/view?usp=sharing
   - Save to: ~/.cache/colorist/diffuseit/id_model/
   - Only needed when use_ffhq=True

4. CLIP Models (Automatic):
   - Multiple CLIP models downloaded automatically via OpenAI CLIP
   - Default: ['RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16']
   - Cached to: ~/.cache/clip/

5. ViT Model (Bundled):
   - Custom ViT model for style/content losses
   - Included in method implementation
   - No manual download required

GPU Requirements:
----------------
- Minimum VRAM: 16GB (for full inference pipeline)
- Recommended: A100 (40GB) or RTX 3090/4090 (24GB)
- Supports float16 for memory efficiency

Manual Setup Required!
---------------------
Unlike StyleID, DiffuseIT requires manual checkpoint downloads:

1. Create checkpoint directory:
   mkdir -p ~/.cache/colorist/diffuseit/

2. Download ImageNet diffusion model from Google Drive (link above)
   
3. Save as: ~/.cache/colorist/diffuseit/256x256_diffusion_uncond.pt

4. (Optional) Download FFHQ model if working with face images

5. Verify checksums (see README.md)

Usage:
------
```python
from code.experiments.reference_methods.style_transfer.training_free.diffuseit.method import DiffuseITMethod

# Initialize method with default parameters
method = DiffuseITMethod(
    checkpoints_dir='~/.cache/colorist/diffuseit',
    device='cuda'
)
method.initialize()  # Loads checkpoints and models

# Inference (image-guided style transfer)
content = torch.randn(1, 3, 256, 256)  # [0, 1] range
style = torch.randn(1, 3, 256, 256)
output = method(content, style)

# Check default configuration
config = method.get_default_config()
print(f"clip_guidance_lambda: {config['clip_guidance_lambda']}")
print(f"timestep_respacing: {config['timestep_respacing']}")
```

Parameter Configuration:
-----------------------
All parameters are defined in method.py via _get_default_config_values().
To customize parameters, edit the _get_default_config_values() method directly.
Default values:

Core Diffusion Parameters (Image-Guided Mode):
- timestep_respacing (str/int): Number of diffusion timesteps
  * Default: "200" (image-guided mode)
  * Text-guided: "100", Image-guided: "200"
  * More steps = better quality, slower inference

- skip_timesteps (int): Steps to skip during diffusion
  * Default: 80 (image-guided mode)
  * Text-guided: 40, Image-guided: 80
  * More skips = faster but lower quality

- diff_iter (int): Number of diffusion iterations
  * Default: 100 (image-guided mode)
  * Text-guided: 50, Image-guided: 100
  * More iterations = better convergence

CLIP Guidance:
- clip_guidance_lambda (float): CLIP guidance strength
  * Default: 2000
  * Range: 1000-5000
  * Higher = stronger style transfer

- clip_models (list): CLIP models to use
  * Default: ['RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16']
  * Options: Single model like ['ViT-B/32'] for memory saving
  * Multiple models = better but slower

ViT Loss Weights:
- vit_lambda (float): ViT contrastive loss weight
  * Default: 300-400
  * Range: 100-1000
  
- lambda_ssim (float): SSIM loss weight
  * Default: 50-100
  * Range: 10-200

Optimization:
- iterations_num (int): Number of optimization iterations
  * Default: 10
  * Range: 5-20
  * More = better quality but much slower

- aug_num (int): Number of augmentations
  * Default: 8
  * Range: 4-16

Model Options:
- use_ffhq (bool): Use FFHQ diffusion model for faces
  * Default: False
  * Set True when processing face images

- use_ddim (bool): Use DDIM instead of DDPM sampling
  * Default: False
  * DDIM is faster but may reduce quality

Method Overview:
---------------
DiffuseIT performs style transfer through CLIP-guided diffusion:

1. Encode content and style images with CLIP (multiple models)
2. Compute target embedding: E_target = E_text - alpha*E_source + beta*E_content
3. Run iterative optimization:
   a. Add noise to content image via diffusion forward process
   b. Predict noise using pretrained UNet
   c. Compute CLIP loss: -similarity(denoised_image, target_embedding)
   d. Compute ViT losses (SSIM, contrastive, content)
   e. Backpropagate through diffusion process
   f. Apply range restart if loss threshold exceeded
4. Denoise to final stylized image

Comparison with StyleID:
------------------------
| Aspect           | StyleID          | DiffuseIT        |
|------------------|------------------|------------------|
| Setup Time       | Minutes (HF)     | Manual (GDrive)  |
| Pretraining      | Not needed       | Not needed       |
| Inference Speed  | ~3-5s            | ~30-60s          |
| GPU Memory       | ~20GB            | ~15-20GB         |
| Quality          | Excellent        | Excellent        |
| Text Guidance    | No               | Yes              |
| Image Guidance   | Yes              | Yes              |

Known Limitations:
-----------------
- Manual checkpoint download required (no automatic setup)
- Very slow inference: ~30-60 seconds per image pair
- Not suitable for training augmentation or large-scale evaluation
- Fixed resolution: 256×256 (interpolated for other sizes)
- Requires multiple CLIP models for best quality (memory intensive)

Performance Notes:
-----------------
- Inference time: ~30-60 seconds per image pair on A100
- GPU memory: ~15-20GB VRAM required
- Quality: State-of-the-art CLIP-guided diffusion
- Best use case: High-quality single-image results, artistic applications

Dependencies:
------------
- torch>=1.9.0
- torchvision>=0.10.0
- clip (from OpenAI GitHub)
- guided-diffusion (custom, bundled in method)
- color-matcher>=0.1.0
- ftfy>=6.0.0
- lpips>=0.1.4
- kornia>=0.6.0

Install with: 
pip install color-matcher ftfy lpips kornia
pip install git+https://github.com/openai/CLIP.git
"""

from .method import Method as DiffuseITMethod

__all__ = ['DiffuseITMethod']
