"""
xAILab Bamberg
University of Bamberg

@description:
StyleID method implementation for training-free style transfer
Based on: https://github.com/jiwoogit/StyleID

Paper: "Style Injection in Diffusion: A Training-free Approach for Adapting 
        Large-scale Diffusion Models for Style Transfer"
Authors: Jiwoo Chung, Sangeek Hyun, Jae-Pil Heo
Conference: CVPR 2024 (Highlight)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler


class StyleIDMethod:
    """
    StyleID (Style Injection in Diffusion) - Training-free style transfer via diffusion.
    
    Paper: "Style Injection in Diffusion: A Training-free Approach for Adapting 
            Large-scale Diffusion Models for Style Transfer"
    Authors: Jiwoo Chung, Sangeek Hyun, Jae-Pil Heo
    Conference: CVPR 2024 (Highlight)
    
    StyleID performs style transfer by:
    1. Encoding content and style images to latent space using VAE
    2. DDIM inversion on both images to extract attention features (K, V)
    3. Optional: AdaIN on content latent using style statistics
    4. DDIM sampling with style attention injection:
       - Use content query (Q) for content preservation
       - Inject style keys (K) and values (V) for style transfer
       - Mix queries with gamma parameter: Q_mix = gamma*Q_c + (1-gamma)*Q_s
       - Scale attention with temperature T
    5. Decode final latent to image space using VAE
    
    This is a TRAINING-FREE method that uses pretrained Stable Diffusion models.
    No pretraining or fine-tuning required.
    """
    
    def __init__(
        self
    ):
        """
        Initialize StyleID method.
        
        This is a training-free method that uses default parameters from get_default_config().
        
        Args:
            pretrained_weights: Unused (kept for interface compatibility). StyleID uses
                              pretrained Stable Diffusion models that are automatically
                              downloaded from HuggingFace.
        """
        # Load default configuration
        default_config = self.get_default_config()
        self.gamma = default_config['gamma']
        self.T = default_config['T']
        self.ddim_steps = default_config['ddim_steps']
        self.sd_version = default_config['sd_version']
        self.injection_layers = default_config['injection_layers']
        self.use_adain = default_config['use_adain']
        self.use_injection = default_config['use_injection']
        self.start_step = default_config['start_step']  # Official: only inject at final steps
        
        # Pretrained model components (loaded in initialize())
        self.pipe = None
        self.vae = None
        self.unet = None
        self.scheduler = None
        self.text_encoder = None
        self.tokenizer = None
        
        # Feature storage for attention injection
        self.content_features = {}
        self.style_features = {}
        self.attn_features_modify = {}  # Features to inject during sampling
        self.hooks = []
        
        # Current timestep tracker for hooks
        self.cur_t = None
        
        self.is_initialized = False
        self.initialize()
    
    def get_default_config(self) -> dict:
        """
        Get default inference configuration.
        
        Note: Training-free methods only have inference parameters.
        No training hyperparameters (lr, batch_size, etc.) needed.
        """
        return {
            'gamma': 0.75,  # Query preservation (0-1). 0.75 = balanced, 0.3 = high style fidelity
            'T': 1.5,  # Attention temperature. Higher = stronger style transfer
            'ddim_steps': 50,  # DDIM sampling steps (official default: 50 in run_styleid.py)
            'sd_version': '1.4',  # Stable Diffusion version (original version sd-v1-4)
            'injection_layers': [6, 7, 8, 9, 10, 11],  # UNet decoder layers for style injection (official: '6,7,8,9,10,11')
            'use_adain': True,  # Apply AdaIN to initial latent
            'use_injection': True,  # Enable attention injection
            'start_step': 49,  # Official: only apply injection from this step onwards (default: 49 of 50)
            'native_image_size': self.get_native_image_size(),  # Image preprocessing
        }
    
    @staticmethod
    def get_native_image_size() -> int:
        """
        Return native image resolution for Stable Diffusion.
        
        Stable Diffusion is trained on 512×512 images (SD 1.x) or 768×768 (SD 2.x).
        We use 512 as the standard for consistency across versions.
        
        Returns:
            512 (native resolution for SD 1.x, also used for SD 2.x for consistency)
        """
        return 512
    
    def initialize(self):
        """
        Load pretrained Stable Diffusion models from HuggingFace.
        
        This replaces load_checkpoint() from training-required methods.
        Models are downloaded automatically on first use and cached locally.
        
        GPU Memory Requirements:
        - Stable Diffusion 1.5: ~5GB VRAM (float16)
        - Full inference pipeline: ~20GB VRAM recommended
        
        Note: Requires internet connection for first-time download.
        """
        if self.is_initialized:
            print("StyleID: Models already initialized")
            return
        
        print("=" * 70)
        print("StyleID: Initializing training-free style transfer method")
        print("=" * 70)
        
        # Map version to HuggingFace model ID (matching official implementation)
        model_id_map = {
            '1.4': 'CompVis/stable-diffusion-v1-4',
            '1.5': 'runwayml/stable-diffusion-v1-5',
            '2.0': 'stabilityai/stable-diffusion-2-base',  # No longer available on HuggingFace
            '2.1': 'stabilityai/stable-diffusion-2-1',  # No longer available on HuggingFace
            '2.1-base': 'stabilityai/stable-diffusion-2-1-base',  # No longer available on HuggingFace
        }
        
        model_id = model_id_map.get(self.sd_version, 'runwayml/stable-diffusion-v1-5')
        print(f"Loading Stable Diffusion {self.sd_version} from HuggingFace...")
        print(f"Model ID: {model_id}")
        
        # Determine dtype (official uses float16 for efficiency)
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Precision: {self.dtype}")
        print(f"Device: {self.device}")
        
        # Load pipeline with DDIM scheduler
        print("Downloading/loading model (this may take a few minutes on first run)...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,  # Disable for speed
            requires_safety_checker=False
        ).to(self.device)
        
        # Replace scheduler with DDIM for inversion
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = self.scheduler
        self.scheduler.set_timesteps(self.ddim_steps)
        
        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        
        # Set to eval mode and disable gradients
        self.vae.eval()
        self.unet.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        
        self.is_initialized = True
        
        print("=" * 70)
        print(f"✓ StyleID initialized successfully")
        print(f"  - Model: Stable Diffusion {self.sd_version}")
        print(f"  - Device: {self.device}")
        print(f"  - Dtype: {self.dtype}")
        print(f"  - Parameters: gamma={self.gamma}, T={self.T}, steps={self.ddim_steps}")
        print(f"  - Injection layers: {self.injection_layers}")
        print(f"  - Start step: {self.start_step}")
        print(f"  - AdaIN enabled: {self.use_adain}")
        print(f"  - Attention injection enabled: {self.use_injection}")
        print("=" * 70)
    
    def __call__(self, content: torch.Tensor, style: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """
        Perform style transfer from style image to content image.
        
        Args:
            content: Content image tensor (B, 3, H, W) in range [0, 1]
            style: Style image tensor (B, 3, H, W) in range [0, 1]
            alpha: Unused (kept for interface compatibility with training-required methods)
            
        Returns:
            Stylized image tensor (B, 3, H, W) in range [0, 1]
            
        Raises:
            RuntimeError: If model not initialized. Call initialize() first.
        """
        if not self.is_initialized:
            raise RuntimeError(
                "StyleID model not initialized. Call method.initialize() first to download models."
            )
        
        # Ensure models are in eval mode
        self.vae.eval()
        self.unet.eval()
        
        # Store original device and dtype for output
        original_device = content.device
        original_dtype = content.dtype
        
        with torch.no_grad():
            # Move inputs to model device and dtype
            content = content.to(device=self.device, dtype=self.dtype)
            style = style.to(device=self.device, dtype=self.dtype)
            
            # Normalize images to [-1, 1] for VAE
            content_norm = (content * 2.0 - 1.0)
            style_norm = (style * 2.0 - 1.0)
            
            # Perform style transfer
            output = self._styleid_transfer(content_norm, style_norm)
            
            # Denormalize back to [0, 1]
            output = (output + 1.0) / 2.0
            
            # Move output back to original device and dtype
            output = output.to(device=original_device, dtype=original_dtype)
        
        return output.clamp(0, 1)
    
    def _styleid_transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Core StyleID algorithm implementation.
        
        Algorithm:
        1. Encode content and style to latent space via VAE
        2. DDIM inversion on both images (get noise + extract attention K, V)
        3. Optional: Apply AdaIN to content latent using style statistics
        4. DDIM sampling with style attention injection:
           - Content provides query (Q) for structure
           - Style provides keys (K) and values (V) for appearance
           - Mix queries: Q_mix = gamma * Q_content + (1-gamma) * Q_stylized
           - Temperature scaling on attention
        5. Decode final latent via VAE
        
        Args:
            content: Content images (B, 3, H, W) in [-1, 1]
            style: Style images (B, 3, H, W) in [-1, 1]
            
        Returns:
            Stylized images (B, 3, H, W) in [-1, 1]
        """
        # Step 1: VAE encode to latent space
        content_latent = self._encode_to_latent(content)
        style_latent = self._encode_to_latent(style)
        
        # Step 2: DDIM inversion to get inverted latents and attention features
        print("StyleID: Inverting content image...")
        content_inverted, self.content_features = self._ddim_inversion(content_latent, extract_features=True)
        
        torch.cuda.empty_cache()

        print("StyleID: Inverting style image...")
        style_inverted, self.style_features = self._ddim_inversion(style_latent, extract_features=True)
        
        # Step 2.5: Prepare attention features for injection (official implementation)
        # Combine content Q with style K, V for each layer and timestep
        self.attn_features_modify = {}
        for layer_name in self.style_features.keys():
            self.attn_features_modify[layer_name] = {}
            for t in self.scheduler.timesteps:
                t_int = int(t.item())
                if t_int in self.content_features[layer_name] and t_int in self.style_features[layer_name]:
                    # Content provides Q, style provides K and V
                    q_content = self.content_features[layer_name][t_int][0]
                    k_style = self.style_features[layer_name][t_int][1]
                    v_style = self.style_features[layer_name][t_int][2]
                    self.attn_features_modify[layer_name][t_int] = (q_content, k_style, v_style)
        
        # Step 3: Optional AdaIN on initial latent
        if self.use_adain:
            initial_latent = self._latent_adain(content_inverted, style_inverted)
        else:
            initial_latent = content_inverted
        
        # Step 4: DDIM sampling with style injection
        print("StyleID: Generating stylized image...")
        stylized_latent = self._ddim_sampling_with_injection(initial_latent)
        
        # Step 5: VAE decode to image space
        stylized = self._decode_from_latent(stylized_latent)
        
        return stylized
    
    def _encode_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to VAE latent space.
        
        Args:
            images: Images (B, 3, H, W) in [-1, 1]
            
        Returns:
            Latents (B, 4, H/8, W/8)
        """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.mode()
        latents = 0.18215 * latents  # Official scaling factor
        return latents
    
    def _decode_from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode VAE latents to image space.
        
        Args:
            latents: Latents (B, 4, H/8, W/8)
            
            Returns:
            Images (B, 3, H, W) in [-1, 1]
        """
        latents = 1 / 0.18215 * latents  # Official scaling factor
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        return images
    
    def _ddim_inversion(
        self, 
        latent: torch.Tensor, 
        extract_features: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Perform DDIM inversion to get noisy latent and optionally extract attention features.
        
        DDIM inversion reverses the diffusion process to find the noise that would
        generate the given latent. During this process, we extract attention K, V
        from specified layers for later style injection.
        
        Args:
            latent: Clean latent (B, 4, H/8, W/8)
            extract_features: Whether to extract attention features for injection
            
        Returns:
            inverted_latent: Noisy latent at t=T (B, 4, H/8, W/8)
            features: Dict of attention features per layer per timestep
        """
        features = {f"layer_{i}_attn": {} for i in self.injection_layers}
        
        # Register hooks to extract attention features
        if extract_features and self.use_injection:
            self._register_extraction_hooks(features)
        
        # Prepare unconditional text embeddings (empty string)
        uncond_embeddings = self._get_text_embeddings("")

        # Repeat embeddings to match batch size
        batch_size = latent.shape[0]
        uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)

        # DDIM inversion: go from clean latent to noisy latent
        # This follows the official implementation's inversion logic
        current_latent = latent.clone()
        timesteps_list = list(reversed(self.scheduler.timesteps))
        num_inference_steps = len(self.scheduler.timesteps)
        
        for i in tqdm(range(num_inference_steps), desc="DDIM Inversion", leave=False):
            t_tensor = timesteps_list[i]
            
            # Set current timestep for hooks
            self.cur_t = t_tensor.item()
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    current_latent,
                    t_tensor,
                    encoder_hidden_states=uncond_embeddings
                ).sample
            
            # Inversion step (official implementation)
            current_t = max(0, t_tensor.item() - (1000 // num_inference_steps))
            next_t = t_tensor.item()
            
            alpha_t = self.scheduler.alphas_cumprod[current_t]
            alpha_t_next = self.scheduler.alphas_cumprod[next_t]
            
            if self.sd_version == "2.1":
                # SD 2.1 inversion (official implementation)
                beta_t = 1 - alpha_t
                pred_original_sample = alpha_t.sqrt() * current_latent - beta_t.sqrt() * noise_pred
                pred_epsilon = alpha_t.sqrt() * noise_pred + beta_t.sqrt() * current_latent
                pred_sample_direction = (1 - alpha_t_next).sqrt() * pred_epsilon
                current_latent = alpha_t_next.sqrt() * pred_original_sample + pred_sample_direction
            else:
                # SD 1.x and 2.1-base inversion (official implementation)
                current_latent = (
                    (current_latent - (1 - alpha_t).sqrt() * noise_pred) * 
                    (alpha_t_next.sqrt() / alpha_t.sqrt()) + 
                    (1 - alpha_t_next).sqrt() * noise_pred
                )
        
        # Remove hooks
        if extract_features and self.use_injection:
            self._remove_hooks()
        
        return current_latent, features
    
    def _ddim_sampling_with_injection(self, noisy_latent: torch.Tensor) -> torch.Tensor:
        """
        DDIM sampling with style attention injection.
        
        During sampling, we inject style attention features:
        - Query (Q): Mix of content and stylized queries (controlled by gamma)
        - Keys (K): From style image
        - Values (V): From style image
        - Temperature (T): Scale attention weights
        
        Args:
            noisy_latent: Starting noisy latent (B, 4, H/8, W/8)
            
        Returns:
            clean_latent: Denoised latent (B, 4, H/8, W/8)
        """
        
        # Register hooks for attention injection
        if self.use_injection:
            self._register_injection_hooks()
        
        # Prepare unconditional text embeddings
        uncond_embeddings = self._get_text_embeddings("")
        
        # Repeat embeddings to match batch size
        batch_size = noisy_latent.shape[0]
        uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)

        # DDIM sampling: go from noisy to clean latent
        current_latent = noisy_latent.clone()
        total_steps = len(self.scheduler.timesteps)
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="DDIM Sampling", leave=False)):
            # Calculate index (official implementation uses reversed index)
            index = total_steps - i - 1
            
            # Official: skip early steps, only apply injection from start_step onwards
            # When index >= start_step, we skip (no injection yet)
            # Injection only happens for index < start_step
            if index >= self.start_step:
                # No injection for early steps - disable injection temporarily
                self.cur_t = None
            else:
                # Set current timestep for hooks to enable injection
                self.cur_t = t.item()
            
            # Prepare input
            latent_model_input = current_latent
            
            # Predict noise with injected style attention
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=uncond_embeddings
                ).sample
            
            # Sampling step
            current_latent = self.scheduler.step(
                noise_pred,
                t,
                current_latent,
                return_dict=False
            )[0]
        
        # Remove hooks
        if self.use_injection:
            self._remove_hooks()
        
        return current_latent
    
    def _latent_adain(self, content_latent: torch.Tensor, style_latent: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN to content latent using style latent statistics.
        
        AdaIN normalizes content latent to have the same mean and std as style latent.
        This helps align the initial latent for better style transfer.
        
        Args:
            content_latent: Content latent (B, 4, H/8, W/8)
            style_latent: Style latent (B, 4, H/8, W/8)
            
        Returns:
            Normalized latent (B, 4, H/8, W/8)
        """
        # Compute statistics (official uses 1e-4 epsilon)
        content_mean = content_latent.mean(dim=[2, 3], keepdim=True)
        content_std = content_latent.std(dim=[2, 3], keepdim=True) + 1e-4
        style_mean = style_latent.mean(dim=[2, 3], keepdim=True)
        style_std = style_latent.std(dim=[2, 3], keepdim=True) + 1e-4
        
        # Normalize and transfer statistics
        normalized = (content_latent - content_mean) / content_std
        output = normalized * style_std + style_mean
        
        return output
    
    def _get_text_embeddings(self, text: str) -> torch.Tensor:
        """
        Get text embeddings for conditioning (used for unconditional = empty string).
        
        Args:
            text: Text prompt (empty string for unconditional)
            
        Returns:
            Text embeddings (1, 77, 768 or 1024)
        """
        tokens = self.tokenizer(
            [text],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        # Get text_encoder's actual device (important when using Accelerate)
        embeddings = self.text_encoder(tokens.input_ids.to(self.text_encoder.device))[0]
        return embeddings
    
    def _register_extraction_hooks(self, features_dict: Dict):
        """
        Register forward hooks on UNet attention layers to extract K, V during inversion.
        
        Args:
            features_dict: Dictionary to store extracted features
        """
        # Get attention layers from UNet decoder
        # SD UNet has up_blocks with attention layers
        attention_layers = self._get_attention_layers()
        
        for layer_idx in self.injection_layers:
            if layer_idx < len(attention_layers):
                layer_name = f"layer_{layer_idx}_attn"
                attn_module = attention_layers[layer_idx]
                
                if attn_module is None:
                    continue
                
                def make_hook(name):
                    def hook(module, input, output):
                        # Extract attention Q, K, V during forward pass
                        # Official implementation extracts from attention_op
                        hidden_states = input[0]
                        
                        # Get Q, K, V (following diffusers attention implementation)
                        query = module.to_q(hidden_states)
                        query = module.head_to_batch_dim(query)
                        
                        encoder_hidden_states = hidden_states
                        key = module.to_k(encoder_hidden_states)
                        key = module.head_to_batch_dim(key)
                        
                        value = module.to_v(encoder_hidden_states)
                        value = module.head_to_batch_dim(value)
                        
                        # Store Q, K, V for this timestep
                        if self.cur_t is not None:
                            features_dict[name][int(self.cur_t)] = (
                                query.detach().cpu(),
                                key.detach().cpu(),
                                value.detach().cpu()
                            )
                    return hook
                
                hook = attn_module.register_forward_hook(make_hook(layer_name))
                self.hooks.append(hook)
    
    def _register_injection_hooks(self):
        """
        Register forward hooks on UNet attention layers to inject style K, V during sampling.
        
        During sampling, hooks will:
        1. Intercept attention computation
        2. Replace content K, V with style K, V
        3. Mix queries: Q_mix = gamma * Q_content + (1-gamma) * Q_current
        4. Apply temperature scaling to attention weights
        """
        attention_layers = self._get_attention_layers()
        
        for layer_idx in self.injection_layers:
            if layer_idx < len(attention_layers):
                layer_name = f"layer_{layer_idx}_attn"
                attn_module = attention_layers[layer_idx]
                
                if attn_module is None:
                    continue
                
                def make_injection_hook(name):
                    def hook(module, input, output):
                        if self.cur_t is None or int(self.cur_t) not in self.attn_features_modify[name]:
                            return output
                        
                        # Get current query, key, value
                        hidden_states = input[0]
                        residual = hidden_states  # For residual connection handling
                        
                        # Handle input dimensions (official implementation)
                        input_ndim = hidden_states.ndim
                        if input_ndim == 4:
                            batch_size, channel, height, width = hidden_states.shape
                            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
                        
                        # Compute current Q, K, V
                        q_cs = module.to_q(hidden_states)
                        q_cs = module.head_to_batch_dim(q_cs)
                        
                        # Get stored content Q and style K, V
                        q_c, k_s, v_s = self.attn_features_modify[name][int(self.cur_t)]
                        
                        q_c = q_c.to(self.device)
                        k_s = k_s.to(self.device)
                        v_s = v_s.to(self.device)
                        # Style injection: Mix queries (official implementation)
                        q_hat_cs = q_c * self.gamma + q_cs * (1 - self.gamma)
                        
                        # Apply temperature scaling to query (official applies to query before attention scores)
                        q_hat_cs = q_hat_cs * self.T
                        
                        # Use style K, V
                        k_cs = k_s
                        v_cs = v_s
                        
                        # Ensure K, V match Q batch size
                        if k_cs.shape[0] != q_hat_cs.shape[0]:
                            k_cs = k_cs[:q_hat_cs.shape[0]]
                            v_cs = v_cs[:q_hat_cs.shape[0]]
                        # Compute attention with injected features (matching official attention_op)
                        # Official uses attn.get_attention_scores which internally scales by sqrt(d_k)
                        scale = 1.0 / (q_hat_cs.shape[-1] ** 0.5)
                        attention_scores = torch.bmm(q_hat_cs, k_cs.transpose(-1, -2)) * scale
                        
                        # Softmax
                        attention_probs = F.softmax(attention_scores, dim=-1)
                        
                        # Apply attention to values
                        hidden_states = torch.bmm(attention_probs, v_cs)
                        hidden_states = module.batch_to_head_dim(hidden_states)
                        
                        # Linear projection
                        hidden_states = module.to_out[0](hidden_states)
                        hidden_states = module.to_out[1](hidden_states)
                        
                        # Handle output dimensions (official implementation)
                        if input_ndim == 4:
                            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
                        
                        # Handle residual connection if enabled (official implementation)
                        if hasattr(module, 'residual_connection') and module.residual_connection:
                            hidden_states = hidden_states + residual
                        
                        # Apply output rescaling if present (official implementation)
                        if hasattr(module, 'rescale_output_factor'):
                            hidden_states = hidden_states / module.rescale_output_factor
                        
                        return hidden_states
                    return hook
                
                hook = attn_module.register_forward_hook(make_injection_hook(layer_name))
                self.hooks.append(hook)
    
    def _get_attention_layers(self) -> List[Optional[nn.Module]]:
        """
        Get attention layers from UNet decoder blocks.
        Matches official implementation's indexing (12 layers total).
        
        Returns:
            List of attention modules (or None for blocks without attention)
        """
        attention_modules = []
        
        # Official implementation assumes 4 up_blocks with 3 layers each = 12 layers
        # Indices 0-2: Block 0 (UpBlock2D - no attention) -> None
        # Indices 3-5: Block 1 (CrossAttnUpBlock2D)
        # Indices 6-8: Block 2 (CrossAttnUpBlock2D)
        # Indices 9-11: Block 3 (CrossAttnUpBlock2D)
        
        if hasattr(self.unet, 'up_blocks'):
            for i in range(12):
                up_block_idx = i // 3
                layer_idx = i % 3
                
                if up_block_idx < len(self.unet.up_blocks):
                    block = self.unet.up_blocks[up_block_idx]
                    
                    if hasattr(block, 'attentions') and layer_idx < len(block.attentions):
                        # This block has attention (CrossAttnUpBlock2D)
                        attn_block = block.attentions[layer_idx]
                        # Get the first transformer block's self-attention (attn1)
                        # Official code: attn[i].transformer_blocks[0].attn1
                        if hasattr(attn_block, 'transformer_blocks'):
                            attention_modules.append(attn_block.transformer_blocks[0].attn1)
                        else:
                            attention_modules.append(None)
                    else:
                        # This block has no attention (UpBlock2D) or layer index out of bounds
                        attention_modules.append(None)
                else:
                    attention_modules.append(None)
        
        return attention_modules
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def eval(self):
        """
        Set model to evaluation mode.
        
        For compatibility with training-required methods that have .eval() method.
        StyleID models are always in eval mode after initialization.
        """
        if self.is_initialized:
            self.vae.eval()
            self.unet.eval()
        return self
    
    def parameters(self):
        """
        Return model parameters.
        
        For compatibility with training-required methods.
        """
        if self.is_initialized:
            import itertools
            return itertools.chain(self.vae.parameters(), self.unet.parameters())
        return iter([])
