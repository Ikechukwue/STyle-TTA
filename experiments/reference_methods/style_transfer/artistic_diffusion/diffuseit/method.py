"""
xAILab Bamberg
University of Bamberg

@description:
DiffuseIT method implementation for training-free style transfer
Based on: https://github.com/cyclomon/DiffuseIT

Paper: "Diffusion-based Image Translation using Disentangled Style
       and Content Representation"
Authors: Gihyun Kwon, Jong Chul Ye
Conference: ICLR 2023

Pretrained weights download:
- 256x256 model trained on ImageNet from: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

# Note: These imports require manual installation
# pip install color-matcher ftfy lpips kornia
# pip install git+https://github.com/openai/CLIP.git
from color_matcher import ColorMatcher

# Manual installation of guided-diffusion is required
# 1. Clone guided-diffusion: git clone https://github.com/openai/guided-diffusion"
# 2. Copy guided_diffusion folder to this directory
from experiments.reference_methods.style_transfer.artistic_diffusion.diffuseit.guided_diffusion.script_util import create_model_and_diffusion

from experiments.reference_methods.style_transfer.artistic_diffusion.diffuseit.utils import Loss_vit, CLIPWrapper, ImageAugmentations


# TensorDict for multi-CLIP model aggregation
# (from official utils_flexit/torch_utils.py)
class TensorDict:
    """Dictionary-like container for tensors that supports element-wise operations."""
    def __init__(self, dct=None, **kwargs):
        self.dct = dct if dct is not None else {}
        self.dct.update(**kwargs)
    
    def __getitem__(self, i):
        if isinstance(i, str) and i in self.dct:
            return self.dct[i]
        elif isinstance(i, TensorDict):
            return TensorDict({k: self.dct[k] * i[k] for k in self.dct})
        else:
            return TensorDict({k: v.__getitem__(i) for k, v in self.dct.items()})
    
    def __getattr__(self, attr):
        if attr == 'dct':
            return super().__getattribute__('dct')
        return TensorDict({k: getattr(v, attr) for k, v in self.dct.items()})
    
    def __call__(self, *args, **kwargs):
        return TensorDict({k: v(*args, **kwargs) for k, v in self.dct.items()})
    
    def __matmul__(self, other):
        if isinstance(other, TensorDict):
            return TensorDict({k: self.dct[k] @ other.dct[k] for k in self.dct})
        return TensorDict({k: v @ other for k, v in self.dct.items()})
    
    def normalize(self):
        return TensorDict({
            k: v / v.norm(dim=-1, keepdim=True) for k, v in self.dct.items()
        })
    
    @property
    def T(self):
        return TensorDict({k: v.T for k, v in self.dct.items()})
    
    def flatten(self):
        return TensorDict({k: v.flatten() for k, v in self.dct.items()})
    
    def reduce(self, fn):
        """Reduce values across all models using fn."""
        values = list(self.dct.values())
        return fn(values)


def mean_sig(x):
    """Mean signature function for aggregating losses."""
    return sum(x) / len(x)


class Method:
    """
    DiffuseIT - Diffusion-based Image Translation.
    
    Uses Disentangled Style and Content Representation.
    
    Paper: "Diffusion-based Image Translation using Disentangled Style
           and Content Representation"
    Authors: Gihyun Kwon, Jong Chul Ye
    Conference: ICLR 2023
    
    DiffuseIT performs style transfer through CLIP-guided iterative optimization:
    1. Encode content and style images with multiple CLIP models
    2. Compute target embedding combining text/style guidance
    3. Run iterative diffusion-based optimization:
       - Forward diffusion: Add noise to content image
       - Reverse diffusion: Predict noise using pretrained UNet
       - CLIP guidance: Optimize toward target embedding
       - ViT losses: Preserve content structure (SSIM, contrastive)
       - Range restart: Reset if loss exceeds threshold
    4. Decode final denoised image
    
    This is a TRAINING-FREE method that uses pretrained diffusion models.
    Models must be manually downloaded from Google Drive.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda'
    ):
        """
        Initialize DiffuseIT method.
        
        This is a training-free method that uses default parameters
        from get_default_config().
        To customize parameters, edit the get_default_config() method or
        pass the parameters to the constructor.
        
        Args:
            checkpoint_path: Path to specific checkpoint file.
                If None, defaults to ~/.cache/colorist/diffuseit
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load default configuration
        default_config = self._get_defaults()
        
        # Core diffusion parameters
        self.timestep_respacing = default_config['timestep_respacing']
        self.skip_timesteps = default_config['skip_timesteps']
        self.diff_iter = default_config['diff_iter']
        self.model_output_size = default_config['model_output_size']
        
        # CLIP guidance parameters
        self.clip_guidance_lambda = default_config['clip_guidance_lambda']
        self.clip_models = default_config['clip_models']
        
        # ViT loss parameters
        self.vit_lambda = default_config['vit_lambda']
        self.lambda_ssim = default_config['lambda_ssim']
        self.lambda_dir_cls = default_config['lambda_dir_cls']
        self.lambda_contra_ssim = default_config['lambda_contra_ssim']
        self.lambda_trg = default_config['lambda_trg']
        
        # L2 and range loss parameters
        self.l2_trg_lambda = default_config['l2_trg_lambda']
        self.range_lambda = default_config['range_lambda']
        
        # Optimization parameters
        self.iterations_num = default_config['iterations_num']
        self.aug_num = default_config['aug_num']
        self.resample_num = default_config['resample_num']
        
        # Model options
        self.use_ffhq = default_config['use_ffhq']
        self.use_ddim = default_config['use_ddim']
        self.use_colormatch = default_config['use_colormatch']
        self.use_range_restart = default_config['use_range_restart']
        self.use_noise_aug_all = default_config['use_noise_aug_all']
        self.use_prog_contrast = default_config['use_prog_contrast']
        self.regularize_content = default_config['regularize_content']
        self.id_lambda = default_config['id_lambda']
        
        # Model components (loaded in initialize())
        self.diffusion_model = None
        self.diffusion = None
        self.clip_net = None  # CLIPS wrapper for multiple CLIP models
        self.clip_models_dict = {}  # Individual CLIP models
        self.vit_loss_model = None
        self.color_matcher = None
        self.image_augmentations = None
        self.idloss = None  # Identity loss for FFHQ mode
        
        self.is_initialized = False
        
        # Initialize models immediately
        self._initialize_network()
    
    @staticmethod
    def _get_defaults() -> dict:
        """
        Get default configuration values for DiffuseIT.
        
        These values are based on the official DiffuseIT repository:
        https://github.com/cyclomon/DiffuseIT
        
        Returns:
            Dictionary with default parameter values
        """
        return {
            # Core diffusion parameters (official defaults from arguments.py)
            'timestep_respacing': '200',  # Number of diffusion steps
            'skip_timesteps': 80,  # Steps to skip during diffusion
            'diff_iter': 100,  # Diffusion iterations for ViT use_dir
            'model_output_size': 256,  # Resolution of diffusion model
            
            # CLIP guidance parameters
            'clip_guidance_lambda': 2000,  # CLIP guidance strength
            'clip_models': [
                'RN50', 'RN50x4', 'ViT-B/32', 'RN50x16', 'ViT-B/16'
            ],  # CLIP models
            
            # ViT loss parameters (matching official arguments.py)
            'vit_lambda': 1,  # ViT total loss multiplier (default: 1)
            'lambda_ssim': 1000,  # Key self similarity loss (default: 1000)
            'lambda_dir_cls': 100,  # Semantic divergence loss (default: 100)
            'lambda_contra_ssim': 200,  # Contrastive loss for keys (default: 200)
            'lambda_trg': 2000,  # Target style loss (default: 2000)
            
            # L2 and range loss
            'l2_trg_lambda': 3000,  # L2 loss for target style image (default: 3000)
            'range_lambda': 200,  # Range loss lambda (default: 200)
            
            # Optimization parameters
            'iterations_num': 10,  # Number of optimization iterations
            'aug_num': 8,  # Number of augmentations
            'resample_num': 10,  # Resampling number (default: 10)
            
            # Model options
            'use_ffhq': False,  # Use FFHQ diffusion model (for faces)
            'use_ddim': False,  # Use DDIM instead of DDPM
            'use_colormatch': True,  # Apply color matching to style
            'use_range_restart': True,  # Restart with high RGB loss
            'use_noise_aug_all': True,  # Use noise augmentation for ViT losses
            'use_prog_contrast': False,  # Progressive contrast
            'regularize_content': False,  # Regularize content when CLIP loss is low
            
            # Identity loss (FFHQ mode)
            'id_lambda': 100,  # Identity loss weight (default: 100)
        }
    
    def get_default_config(self) -> dict:
        """
        Get default inference configuration.
        
        Note: Training-free methods only have inference parameters.
        No training hyperparameters (lr, batch_size, etc.) needed.
        """
        return {
            # Method identification
            'method_type': 'training-free',
            'method_name': 'diffuseit',
            
            # Core parameters
            'timestep_respacing': self.timestep_respacing,
            'skip_timesteps': self.skip_timesteps,
            'diff_iter': self.diff_iter,
            'model_output_size': self.model_output_size,
            'clip_guidance_lambda': self.clip_guidance_lambda,
            'clip_models': self.clip_models,
            'vit_lambda': self.vit_lambda,
            'lambda_ssim': self.lambda_ssim,
            'lambda_dir_cls': self.lambda_dir_cls,
            'lambda_contra_ssim': self.lambda_contra_ssim,
            'lambda_trg': self.lambda_trg,
            'l2_trg_lambda': self.l2_trg_lambda,
            'range_lambda': self.range_lambda,
            'iterations_num': self.iterations_num,
            'aug_num': self.aug_num,
            'resample_num': self.resample_num,
            'use_ffhq': self.use_ffhq,
            'use_ddim': self.use_ddim,
            'use_colormatch': self.use_colormatch,
            'use_range_restart': self.use_range_restart,
            'use_noise_aug_all': self.use_noise_aug_all,
            'use_prog_contrast': self.use_prog_contrast,
            'regularize_content': self.regularize_content,
            'id_lambda': self.id_lambda,
            
            # Image processing
            'native_image_size': self.get_native_image_size(),
            
            # Hardware
            'device': self.device,
        }
    
    @staticmethod
    def get_native_image_size() -> int:
        """
        Get the native image size for the diffusion model.
        
        DiffuseIT uses 256×256 resolution diffusion models.
        Images of other sizes are resized to 256×256 for processing.
        
        Returns:
            Native image size (256)
        """
        return 256
    
    def _initialize_network(self):
        """
        Load pretrained diffusion models and CLIP models.
        
        This replaces load_checkpoint() from training-required methods.
        
        Models are loaded from:
        1. Diffusion model: Manual download from Google Drive (ImageNet or FFHQ)
        2. CLIP models: Automatic download from OpenAI
        3. LPIPS: Automatic download from PyTorch Hub
        
        GPU Memory Requirements:
        - Diffusion model: ~5GB
        - Multiple CLIP models: ~3-5GB
        - Full inference: ~15-20GB VRAM recommended
        
        Raises:
            FileNotFoundError: If diffusion checkpoint not found
            RuntimeError: If model loading fails
        """
        if self.is_initialized:
            print("DiffuseIT already initialized, skipping...")
            return
        
        print("=" * 70)
        print("DiffuseIT: Initializing training-free style transfer method")
        print("=" * 70)

        # Check if checkpoint exists
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Please create directory and download diffusion models:\n"
                f"  mkdir -p {self.checkpoint_path}\n"
                f"  # Download ImageNet model from:\n"
                f"  # https://drive.google.com/file/d/"
                f"1kfCPMZLaAcpoIcvzTHwVVJ_qDetH-Rns/view\n"
                f"  # Save as: {self.checkpoint_path}/diffuseit.pt"
            )
        
        print(f"Loading diffusion model from: {self.checkpoint_path}")
        
        # Load diffusion model
        # Note: This is a placeholder - actual implementation would use guided-diffusion
        # For now, we'll create a dummy model to demonstrate the structure
        try:
            self.diffusion_model = self._load_diffusion_model(self.checkpoint_path)
            print(f"✓ Diffusion model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load diffusion model: {e}")
        
        # Load CLIP models and create CLIPS wrapper
        print(f"Loading {len(self.clip_models)} CLIP models...")
        for model_name in self.clip_models:
            try:
                model = CLIPWrapper(model_name, device=self.device, erasing=False)
                self.clip_models_dict[model_name] = model
                print(f"  ✓ Loaded CLIP model: {model_name}")
            except Exception as e:
                print(f"  ✗ Failed to load CLIP model {model_name}: {e}")
                # Continue with other models
        
        if not self.clip_models_dict:
            raise RuntimeError("Failed to load any CLIP models")
        
        # Create CLIPS wrapper (aggregates multiple CLIP models)
        self.clip_net = self._create_clips_wrapper()
        print(f"✓ CLIPS wrapper created with {len(self.clip_models_dict)} models")
        
        # Initialize image augmentations for CLIP
        self.image_augmentations = ImageAugmentations(
            output_size=224,  # CLIP input size
            augmentations_number=self.aug_num
        ).to(self.device)
        print("✓ Image augmentations initialized")
        
        # Load ViT Loss model (DINO)
        print("Loading ViT Loss model (DINO)...")
        self.vit_loss_model = Loss_vit(
            device=self.device,
            lambda_ssim=self.lambda_ssim,
            lambda_dir_cls=self.lambda_dir_cls,
            lambda_contra_ssim=self.lambda_contra_ssim,
            lambda_trg=self.lambda_trg
        ).to(self.device)
        self.vit_loss_model.eval()
        print("✓ ViT Loss model loaded (DINO vits16)")
        
        # Initialize identity loss for FFHQ mode
        if self.use_ffhq:
            try:
                # IDLoss would need to be imported/implemented
                # For now, we'll set it to None and handle in _cond_fn
                print("Note: FFHQ mode - identity loss not yet implemented")
                self.idloss = None
            except Exception as e:
                print(f"Warning: Could not load identity loss: {e}")
                self.idloss = None
        
        # Initialize color matcher if enabled
        if self.use_colormatch:
            self.color_matcher = ColorMatcher()
            print("✓ Color matcher initialized")
        
        self.is_initialized = True
        
        print("=" * 70)
        print(f"✓ DiffuseIT initialized successfully")
        print(f"  - Model: ImageNet 256×256 Diffusion")
        print(f"  - Device: {self.device}")
        print(f"  - CLIP models: {len(self.clip_models_dict)} loaded")
        print(f"  - Parameters: clip_lambda={self.clip_guidance_lambda}, "
              f"vit_lambda={self.vit_lambda}, iterations={self.iterations_num}")
        print("=" * 70)
    
    def _load_diffusion_model(self, checkpoint_path: Path):
        """
        Load pretrained diffusion model from checkpoint.
        
        Loads the guided-diffusion UNet model and configures the diffusion scheduler.
        
        Args:
            checkpoint_path: Path to diffusion model checkpoint
            
        Returns:
            Loaded diffusion model (UNet)
        """
        
        # Load checkpoint
        print(f"  Loading checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Configure model based on checkpoint type (ImageNet vs FFHQ)
        if self.use_ffhq:
            # FFHQ 256x256 model configuration (matching official)
            model_config = {
                'image_size': 256,
                'num_channels': 256,
                'num_res_blocks': 2,
                'num_heads': 4,
                'num_heads_upsample': -1,
                'num_head_channels': -1,
                'attention_resolutions': '16',  # Official FFHQ uses only 16
                'channel_mult': '',
                'dropout': 0.0,
                'class_cond': False,
                'use_checkpoint': False,
                'use_scale_shift_norm': True,
                'resblock_updown': True,
                'use_fp16': False,
                'use_new_attention_order': False,
                'learn_sigma': True,
                'diffusion_steps': 1000,
                'noise_schedule': 'linear',
                'timestep_respacing': self.timestep_respacing,
                'use_kl': False,
                'predict_xstart': False,
                'rescale_timesteps': False,
                'rescale_learned_sigmas': False,
            }
        else:
            # ImageNet 256x256 model configuration
            model_config = {
                'image_size': 256,
                'num_channels': 256,
                'num_res_blocks': 2,
                'num_heads': 4,
                'num_heads_upsample': -1,
                'num_head_channels': -1,
                'attention_resolutions': '32,16,8',
                'channel_mult': '',
                'dropout': 0.0,
                'class_cond': False,  # ImageNet unconditional model
                'use_checkpoint': False,
                'use_scale_shift_norm': True,
                'resblock_updown': True,
                'use_fp16': False,
                'use_new_attention_order': False,
                'learn_sigma': True,
                'diffusion_steps': 1000,
                'noise_schedule': 'linear',
                'timestep_respacing': self.timestep_respacing,
                'use_kl': False,
                'predict_xstart': False,
                'rescale_timesteps': False,
                'rescale_learned_sigmas': False,
            }
        
        # Store model config for later use
        self.model_config = model_config
        
        # Create model and diffusion
        print(f"  Creating UNet model...")
        model, diffusion = create_model_and_diffusion(**model_config)
        
        # Load weights
        print(f"  Loading model weights...")
        model.load_state_dict(checkpoint)
        
        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()
        
        # Disable all gradients first, then enable for specific layers
        # This matches official implementation:
        # model.requires_grad_(False).eval()
        # for name, param in model.named_parameters():
        #     if "qkv" in name or "norm" in name or "proj" in name:
        #         param.requires_grad_()
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_(True)
        
        # Convert to fp16 if configured
        if self.model_config.get('use_fp16', False):
            model.convert_to_fp16()
        
        # Store diffusion scheduler
        self.diffusion = diffusion
        
        print(f"  ✓ Diffusion model loaded successfully")
        return model
    
    def _create_clips_wrapper(self):
        """
        Create a CLIPS-like wrapper that aggregates multiple CLIP models.
        Returns a wrapper object with encode_image and encode_text methods.
        """
        class CLIPS:
            def __init__(self, networks):
                self.networks = networks
            
            def encode_image(self, x, ncuts=0):
                return TensorDict({name: model.encode_image(x, ncuts=ncuts) 
                                   for name, model in self.networks.items()})
            
            def encode_text(self, x):
                return TensorDict({name: model.encode_text(x) 
                                   for name, model in self.networks.items()})
        
        return CLIPS(self.clip_models_dict)
    
    def __call__(
        self, 
        content: torch.Tensor, 
        style: torch.Tensor, 
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Perform style transfer from style image to content image.
        
        Args:
            content: Content image tensor (B, 3, H, W) in range [0, 1]
            style: Style image tensor (B, 3, H, W) in range [0, 1]
            alpha: Unused (kept for interface compatibility)
            
        Returns:
            Stylized image tensor (B, 3, H, W) in range [0, 1]
            
        Raises:
            RuntimeError: If model not initialized. Call initialize() first.
        """
        if not self.is_initialized:
            raise RuntimeError(
                "DiffuseIT not initialized. Call initialize() first to load models."
            )
        
        # Ensure models are in eval mode
        for model in self.clip_models_dict.values():
            model.eval()
        self.vit_loss_model.eval()
        
        with torch.no_grad():
            # Resize to native size if needed
            content_resized = self._resize_to_native(content)
            style_resized = self._resize_to_native(style)
            
            # Perform DiffuseIT style transfer
            output = self._diffuseit_transfer(content_resized, style_resized)
            
            # Resize back to original size if needed
            if output.shape[-2:] != content.shape[-2:]:
                output = F.interpolate(
                    output, 
                    size=content.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
        
        return output.clamp(0, 1)
    
    def _resize_to_native(self, img: torch.Tensor) -> torch.Tensor:
        """Resize image to native diffusion model size (256×256)."""
        native_size = self.get_native_image_size()
        if img.shape[-2:] != (native_size, native_size):
            return F.interpolate(
                img, 
                size=(native_size, native_size), 
                mode='bilinear', 
                align_corners=False
            )
        return img
    
    def _diffuseit_transfer(
        self, 
        content: torch.Tensor, 
        style: torch.Tensor
    ) -> torch.Tensor:
        """
        Core DiffuseIT style transfer algorithm.
        Faithful implementation of edit_image_by_prompt from official repo.
        
        Supports both image-guided and text-guided modes:
        - Image-guided: style is a tensor (uses target_image path)
        - Text-guided: would use prompt/source text (not directly supported here)
        """
        # Convert images to [-1, 1] for diffusion (official format)
        self.init_image = content * 2 - 1
        self.target_image = style * 2 - 1
        self.prev = self.init_image.detach()
        self.loss_prev = torch.tensor(0.0).to(self.device)
        self.flag_resample = False
        
        # For text-guided mode (if target_image is None), compute target embedding
        # self.tgt = (1 * E_T - 0.4 * E_S + 0.2 * E_I0).normalize()
        # Since we're doing image-guided, we skip this
        self.tgt = None
        
        # Check if diffusion model is properly initialized
        if not hasattr(self, 'diffusion') or self.diffusion is None:
            raise RuntimeError("Diffusion model not initialized")
            
        # Main optimization loop
        total_steps = self.diffusion.num_timesteps - self.skip_timesteps - 1
        total_steps_with_resample = total_steps + (self.resample_num - 1)
        final_sample = None
        
        for iteration_number in range(self.iterations_num):
            print(f"Start iteration {iteration_number}")
            
            # Reset resample flag
            self.flag_resample = False
            
            # Choose sampling function (DDIM or DDPM)
            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.use_ddim
                else self.diffusion.p_sample_loop_progressive
            )
            
            # Create noisy version of init_image to start the reverse process
            # This replaces the init_image parameter from the original repo
            # which was added to their modified guided_diffusion
            start_timestep = self.diffusion.num_timesteps - self.skip_timesteps - 1
            noise = torch.randn_like(self.init_image)
            t_start = torch.tensor([start_timestep], device=self.device)
            x_noisy = self.diffusion.q_sample(
                x_start=self.init_image,
                t=t_start,
                noise=noise
            )
            
            # Run diffusion loop with guided sampling
            samples = sample_func(
                self.diffusion_model,
                (
                    1,  # Batch size
                    3,
                    self.model_output_size,
                    self.model_output_size,
                ),
                noise=x_noisy,  # Start from noisy init_image
                clip_denoised=False,
                model_kwargs={},  # No y for unconditional model
                cond_fn=self._cond_fn,
                progress=True,
                skip_timesteps=self.skip_timesteps,
            )
            
            # Iterate through samples
            for j, sample in enumerate(samples):
                final_sample = sample["pred_xstart"]
                
                # Check for range restart (handled via flag set in cond_fn)
                if self.flag_resample:
                    break
            
            if self.flag_resample:
                print(f"  Range restart triggered at iteration {iteration_number}")
                continue
        
        if final_sample is None:
            return content  # Should not happen
            
        # Convert back to [0, 1]
        result = (final_sample + 1) / 2
        
        # Apply color matching (Post-processing)
        if self.use_colormatch and self.color_matcher is not None:
            result = self._apply_color_matching(result, style)
        
        return result.clamp(0, 1)

    def _cond_fn(self, x, t, y=None):
        """
        Conditioning function for guided diffusion.
        Calculates gradients based on CLIP and ViT losses.
        
        Matches official implementation from image_editor.py cond_fn.
        """
        self.flag_resample = False
        
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            
            # t_unscaled: maps spaced index → 0-999 scale for use_dir / diff_iter comparisons
            # and for p_mean_variance (which expects 0-999 timesteps).
            t_unscaled = self._unscale_timestep(t)
            # t_spaced: raw spaced index (0..num_timesteps-1) for _noisy_aug array indexing.
            # sqrt_one_minus_alphas_cumprod has num_timesteps (e.g. 200) elements, so we
            # must index with the spaced value, NOT the 0-999 unscaled value.
            t_spaced = int(t[0].item())
            
            # Get prediction from diffusion model
            # Note: model_kwargs is empty for unconditional model
            out = self.diffusion.p_mean_variance(
                self.diffusion_model, x, t_unscaled, clip_denoised=False, model_kwargs={}
            )
            
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Determine frac_cont for progressive contrast
            frac_cont = 1.0
            if self.target_image is None:  # Text-guided mode
                if self.use_prog_contrast:
                    if self.loss_prev > -0.5:
                        frac_cont = 0.5
                    elif self.loss_prev > -0.4:
                        frac_cont = 0.25
                if self.regularize_content:
                    if self.loss_prev < -0.5:
                        frac_cont = 2
            
            # 1. CLIP Loss (Only for text-guided mode, i.e., target_image is None)
            if self.target_image is None and self.clip_guidance_lambda != 0:
                # Apply noisy augmentation for CLIP (use spaced index for array lookup)
                x_clip = self._noisy_aug(t_spaced, x, out["pred_xstart"])
                # Encode with CLIP (convert to [0, 1] range)
                pred = self.clip_net.encode_image(0.5 * x_clip + 0.5, ncuts=self.aug_num)
                # Compute CLIP loss against target embedding
                if self.tgt is not None:
                    clip_loss = -(pred @ self.tgt.T).flatten().reduce(mean_sig)
                    loss = loss + clip_loss * self.clip_guidance_lambda
                    self.loss_prev = clip_loss.detach().clone()
            
            # 2. ViT Loss
            # Use noisy augmentation if enabled (use spaced index for array lookup)
            if self.use_noise_aug_all:
                x_in = self._noisy_aug(t_spaced, x, out["pred_xstart"])
            else:
                x_in = out["pred_xstart"]
            
            # Calculate ViT loss
            if self.vit_lambda != 0:
                use_dir = (t_unscaled[0] > self.diff_iter)
                
                vit_loss, _ = self.vit_loss_model(
                    x_in, 
                    self.init_image, 
                    self.prev, 
                    use_dir=use_dir, 
                    frac_cont=frac_cont, 
                    target=self.target_image
                )
                loss = loss + vit_loss * self.vit_lambda
            
            # 3. Range Loss
            r_loss_val = torch.tensor(0.0, device=self.device)
            if self.range_lambda != 0:
                # Official: range_loss(input) = (input - input.clamp(-1, 1)).pow(2).mean([1,2,3])
                r_loss_val = (out["pred_xstart"] - out["pred_xstart"].clamp(-1, 1)).pow(2).mean()
                loss = loss + r_loss_val * self.range_lambda
            
            # 4. MSE Loss for Image-Guided mode
            if self.target_image is not None:
                loss = loss + F.mse_loss(x_in, self.target_image) * self.l2_trg_lambda
            
            # 5. Identity Loss for FFHQ mode
            if self.use_ffhq and self.idloss is not None:
                loss = loss + self.idloss(x_in, self.init_image) * self.id_lambda
            
            # Update prev for next iteration
            self.prev = x_in.detach().clone()
            
            # Check for range restart
            if self.use_range_restart:
                total_steps = self.diffusion.num_timesteps - self.skip_timesteps - 1
                if t_unscaled[0].item() < total_steps:
                    threshold = 0.1 if self.use_ffhq else 0.01
                    if r_loss_val.item() > threshold:
                        self.flag_resample = True
            
            return -torch.autograd.grad(loss, x)[0]

    def _unscale_timestep(self, t):
        # t is a spaced index in [0, num_timesteps-1] (e.g. 0-199 for 200-step diffusion).
        # Map it to the 0-999 scale used by diffusion internals and for use_dir comparisons.
        # Original (wrong): t * (num_timesteps / 1000) → 0-199 * 0.2 = 0-39 (always < diff_iter=100)
        # Fixed: t * (1000 / num_timesteps) → 0-199 * 5 = 0-995 (correctly spans 0-999)
        unscaled_timestep = (t * (1000 / self.diffusion.num_timesteps)).long()
        return unscaled_timestep

    def _noisy_aug(self, t, x, x_hat):
        fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t]
        x_mix = x_hat * fac + x * (1 - fac)
        return x_mix

    
    def _apply_color_matching(
        self, 
        content: torch.Tensor, 
        style: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply color matching from style to content (result).
        
        Args:
            content: Image to modify (Result) (B, 3, H, W)
            style: Reference image (Style) (B, 3, H, W)
            
        Returns:
            Color-matched image
        """
        # Convert to numpy for color-matcher library
        content_np = content[0].permute(1, 2, 0).cpu().numpy()
        style_np = style[0].permute(1, 2, 0).cpu().numpy()
        
        # Apply color matching
        # Match content (result) to style colors
        matched_np = self.color_matcher.transfer(
            src=content_np,
            ref=style_np,
            method='mkl'  # Use Monge-Kantorovich method
        )
        
        # MKL eigendecomposition can yield complex values; take the real part
        if np.iscomplexobj(matched_np):
            matched_np = matched_np.real
        matched_np = matched_np.astype(np.float32)

        # Convert back to tensor
        matched = torch.from_numpy(matched_np).permute(2, 0, 1).unsqueeze(0)
        return matched.to(style.device)
    
    def eval(self):
        """
        Set all models to evaluation mode.
        
        This disables dropout, batch norm updates, etc.
        """
        if self.diffusion_model is not None:
            self.diffusion_model.eval()
        
        for model in self.clip_models_dict.values():
            model.eval()
        
        if self.vit_loss_model is not None:
            self.vit_loss_model.eval()
    
    def parameters(self):
        """
        Get all model parameters.
        
        Returns empty list since this is a training-free method
        with frozen pretrained models.
        """
        return []
