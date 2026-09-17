"""
xAILab Bamberg
University of Bamberg

@description:
DiffStyle method implementation for training-free style transfer
Based on: https://github.com/curryjung/InjectFusion_official

Paper: "Training-free Content Injection using h-space in Diffusion models"
Authors: Jaeseok Jeong, Mingi Kwon, Youngjung Uh

Pretrained weights download:
- 256x256 model trained on ImageNet from: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from code.experiments.reference_methods.style_transfer.artistic_diffusion.diffstyle.models.improved_ddpm.script_util import (
    create_model,
)
from code.experiments.reference_methods.style_transfer.artistic_diffusion.diffstyle.utils.diffusion_utils import (
    get_beta_schedule,
    denoising_step,
)


class Method:
    def __init__(self, model_path=None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = 256
        self.num_channels = 256
        self.num_res_blocks = 2
        self.num_heads = 4
        self.num_heads_upsample = -1
        self.attention_resolutions = "32,16,8"
        self.dropout = 0.0
        self.learn_sigma = True
        self.sigma_small = False
        self.class_cond = False
        self.diffusion_steps = 1000
        self.noise_schedule = "linear"
        self.timestep_respacing = ""
        self.use_kl = False
        self.predict_xstart = False
        self.rescale_timesteps = True
        self.rescale_learned_sigmas = True
        self.use_checkpoint = False
        self.use_scale_shift_norm = True

        self.model = None
        self.config = self.get_default_config()

        if model_path is None:
            model_path = "/data/local/colorist/models/pretrained/diffstyle.pt"
        
        self._initialize_network(model_path)

    def _initialize_network(self, model_path: str):
        self.model = create_model(
            self.image_size,
            self.num_channels,
            self.num_res_blocks,
            channel_mult="1,1,2,2,4,4",
            learn_sigma=self.learn_sigma,
            class_cond=self.class_cond,
            use_checkpoint=self.use_checkpoint,
            attention_resolutions=self.attention_resolutions,
            num_heads=self.num_heads,
            num_head_channels=self.num_heads_upsample,
            num_heads_upsample=self.num_heads_upsample,
            use_scale_shift_norm=self.use_scale_shift_norm,
            dropout=self.dropout,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
        )

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.betas = get_beta_schedule(
            beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=self.diffusion_steps
        )
        self.betas = torch.from_numpy(self.betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1.0 - self.betas
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_default_config(self):
        return {
            "t_edit": 400,  # Timestep threshold for h-space editing
            "hs_coeff": 0.4,  # Slerp ratio (h_gamma in official) - 0.4 for cross-domain
            "n_inv_step": 1000,  # Number of inversion steps
            "n_gen_step": 1000,  # Number of generation steps
            "t_noise": 0,  # Quality boosting threshold (t_boost) - 0 for out-of-domain
            "dt_lambda": 1.0,  # Sampling calibration factor (0.9985 for masked style mixing)
            "dt_end": 950,  # dt_lambda application threshold
            "omega": 0.0,  # Style calibration parameter
        }

    def get_native_image_size(self):
        """Return the native image size expected by the model."""
        return self.image_size

    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            # Resize if needed
            if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
                image = torch.nn.functional.interpolate(
                    image, size=(self.image_size, self.image_size), mode="bilinear"
                )
            # Normalize to [-1, 1] if in [0, 1]
            if image.min() >= 0 and image.max() <= 1:
                image = image * 2 - 1
            return image.to(self.device)
        elif isinstance(image, Image.Image):
            return self.transform(image).unsqueeze(0).to(self.device)
        else:
            raise ValueError("Unsupported image type")

    def postprocess(self, tensor):
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
        return tensor

    def __call__(self, content, style, alpha=None):
        config = self.get_default_config()
        t_edit = config["t_edit"]
        hs_coeff_val = config["hs_coeff"]
        n_inv_step = config["n_inv_step"]
        n_gen_step = config["n_gen_step"]
        t_noise = config["t_noise"]
        dt_lambda = config["dt_lambda"]
        dt_end = config["dt_end"]
        omega = config["omega"]
        t_0 = self.num_timesteps - 1  # 999

        content_tensor = self.preprocess(content)
        style_tensor = self.preprocess(style)
        
        batch_size = content_tensor.shape[0]
        # Use class 0 as dummy class if conditional
        y = torch.zeros(batch_size, dtype=torch.long, device=self.device) if self.class_cond else None

        # Build inversion sequence (coarse - fewer steps)
        seq_inv = np.linspace(0, 1, n_inv_step) * t_0
        seq_inv = [int(s + 1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])
        
        # Build generation sequence (fine - more steps)
        seq_gen = np.linspace(0, 1, n_gen_step) * t_0
        seq_gen = [int(s + 1e-6) for s in list(seq_gen)]
        seq_gen_next = [-1] + list(seq_gen[:-1])
        
        # Store timesteps for h-feature lookup
        inv_timesteps = sorted(list(set(seq_inv)), reverse=True)
        
        def take_closest(input_list, value):
            """Return closest value among input_list."""
            return min(input_list, key=lambda x: abs(x - value))
        
        # ============ INVERT CONTENT and store h-features ============
        # Official: precompute_pairs_with_h stores content h-features
        xt = content_tensor.clone()
        content_h_dict = {}  # Store content h for each timestep
        
        for it, (i, j) in enumerate(tqdm(zip(seq_inv_next[1:], seq_inv[1:]), 
                                          total=len(seq_inv)-1, desc="Inverting Content")):
            t = (torch.ones(batch_size) * i).to(self.device)
            t_prev = (torch.ones(batch_size) * j).to(self.device)
            
            # Extract h-space features during inversion
            et, _, _, h = self.model(xt, t, y=y)
            if self.learn_sigma:
                et, _ = torch.split(et, et.shape[1] // 2, dim=1)
            
            # Store h for this timestep
            content_h_dict[int(i)] = h.detach().clone()
            
            at = self.alphas_cumprod[i]
            at_next = self.alphas_cumprod[j]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            xt = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            
        # xt is now the inverted content latent (not used for generation)
        
        # ============ INVERT STYLE to get style latent X_T ============
        xt = style_tensor.clone()
        
        for it, (i, j) in enumerate(tqdm(zip(seq_inv_next[1:], seq_inv[1:]), 
                                          total=len(seq_inv)-1, desc="Inverting Style")):
            t = (torch.ones(batch_size) * i).to(self.device)
            t_prev = (torch.ones(batch_size) * j).to(self.device)
            
            et, _, _, _ = self.model(xt, t, y=y)
            if self.learn_sigma:
                et, _ = torch.split(et, et.shape[1] // 2, dim=1)
            
            at = self.alphas_cumprod[i]
            at_next = self.alphas_cumprod[j]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            xt = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            
        X_T_style = xt.clone()  # Inverted style latent - this is what we generate from
        
        # ============ GENERATION: Generate from style X_T with content h-injection ============
        # Official: diff_style generates from style X_T, injecting content h-features
        seq_gen_reversed = list(reversed(seq_gen))
        seq_gen_next_reversed = list(reversed(seq_gen_next))
        
        x = X_T_style.clone()  # Start from style latent
        
        for i, j in tqdm(zip(seq_gen_reversed, seq_gen_next_reversed), 
                         total=len(seq_gen_reversed), desc="Stylizing"):
            t = (torch.ones(batch_size) * i).to(self.device)
            t_next = (torch.ones(batch_size) * j).to(self.device)
            
            # Find closest timestep in content_h_dict for h-injection
            closest_t = take_closest(list(content_h_dict.keys()), int(i))
            content_h = content_h_dict[closest_t].to(self.device)
            
            # Determine eta for quality boosting
            # Official: eta=0 when t > t_noise, eta=1 when t <= t_noise
            eta = 1.0 if int(i) <= t_noise else 0.0
            
            # Denoise with content h-injection
            # Official: pass --hs_coeff $h_gamma directly (0.4 for cross-domain)
            # UNet slerp: slerp(1 - hs_coeff[0], h_style, h_content)
            # hs_coeff[0]=0.4 → slerp(0.6, h_style, h_content) = 60% toward content ✓
            x, _, _, _ = denoising_step(
                x, t, t_next,
                models=self.model,
                logvars=None,
                b=self.betas,
                sampling_type="ddim",
                eta=eta,
                learn_sigma=self.learn_sigma,
                index=0,  # Use index 0 for injection
                t_edit=t_edit,
                hs_coeff=[hs_coeff_val],  # Direct: matches official --hs_coeff $h_gamma
                delta_h=content_h,
                dt_lambda=dt_lambda,
                dt_end=dt_end,
                omega=omega,
                y=y
            )
            
        return self.postprocess(x)

