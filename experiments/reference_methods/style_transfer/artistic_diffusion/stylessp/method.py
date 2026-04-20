import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from typing import Optional, Dict, Tuple, List
from transformers import AutoProcessor, Blip2ForConditionalGeneration, DPTForDepthEstimation, DPTFeatureExtractor, CLIPVisionModelWithProjection
from diffusers import DDIMScheduler, UniPCMultistepScheduler, AutoencoderKL, ControlNetModel
from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel
from diffusers.utils import load_image

from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.src.config import RunConfig
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.src.eunms import Model_Type, Scheduler_Type
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.src.utils.enums_utils import get_pipes
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.src.frequency_utils import freq_exp
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.inversion import run as invert
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.pipeline_stable_diffusion_sdxl_extra_cfg import StableDiffusionXLPipelineExtraCFG
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.pipeline_stable_diffusion_extra_cfg import StableDiffusionPipelineCFG
from experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.ip_adapter_instruct import IPAdapterInstructSDXL, IPAdapterInstruct

class Method:
    """
    StyleSSP - Style Transfer via Style-Structure Preserving (Training-free)
    """
    
    def __init__(self, pretrained_weights: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        
        # Parse pretrained_weights for IP-Adapter paths
        self.ip_adapter_path = None
        self.ip_instruct_ckpt = None
        
        if pretrained_weights:
            if isinstance(pretrained_weights, str):
                if ',' in pretrained_weights:
                    paths = pretrained_weights.split(',')
                    if len(paths) >= 2:
                        self.ip_adapter_path = paths[0].strip()
                        self.ip_instruct_ckpt = paths[1].strip()
                else:
                    # If only one path provided, assume it's the adapter path
                    self.ip_adapter_path = pretrained_weights
            elif isinstance(pretrained_weights, (list, tuple)):
                if len(pretrained_weights) >= 2:
                    self.ip_adapter_path = pretrained_weights[0]
                    self.ip_instruct_ckpt = pretrained_weights[1]
        
        # Fallback to env vars if not set
        if not self.ip_adapter_path:
             self.ip_adapter_path = os.environ.get("IP_ADAPTER_PATH_DIR", "./checkpoints/IP-Adapter")
        if not self.ip_instruct_ckpt:
             self.ip_instruct_ckpt = os.environ.get("IP_INSTRUCT_CKPT", "./checkpoints/models/ip-adapter-instruct-sdxl.bin")
        
        # Load default configuration
        self.config_dict = self.get_default_config()
        
        # Initialize model placeholders
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.image_encoder = None
        self.blip_model = None
        self.blip2_model = None
        self.pipe_inference = None
        self.ip_instruct_model = None
        self.blip_processor = None
        self.depth_estimator = None
        self.depth_feature_extractor = None
        
        self.is_initialized = False
        self._initialize_network()

    def get_default_config(self) -> dict:
        """
        Get default inference configuration.
        Matches the official infer_style.py overrides of RunConfig defaults.
        """
        return {
            'method_type': 'training-free',
            'method_name': 'stylessp',
            'resolution': 1024,
            'seed': 7865,
            'model_type': Model_Type.SDXL,
            'scheduler_type': Scheduler_Type.DDIM,
            'num_inference_steps': 50,
            'num_inversion_steps': 50,
            'num_renoise_steps': 1,
            'perform_noise_correction': False,
            'guidance_scale': 5.0,
            'style_guidance_scale': 0.0,
            'content_guidance_scale': 0.0,
            'inv_guidance': 0.9,
            'control_type': 'tile_canny', 
            'base_model_path': "stabilityai/stable-diffusion-xl-base-1.0",
            'IP_path': self.ip_adapter_path,
            'ip_ckpt': self.ip_instruct_ckpt,
            'tile_controlnet_path': "xinsir/controlnet-tile-sdxl-1.0",
            'canny_controlnet_path': "TheMistoAI/MistoLine",
            'depth_controlnet_path': "diffusers/controlnet-depth-sdxl-1.0-small",
            'device': self.device,
            'choose_pipeline': 'sdxl',
            'native_image_size': self.get_native_image_size(),
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 1024

    def _initialize_network(self):
        if self.is_initialized:
            return

        print("=" * 70)
        print("StyleSSP: Initializing training-free style transfer method")
        print("=" * 70)

        # Create RunConfig object from config_dict
        self.run_config = RunConfig()
        for k, v in self.config_dict.items():
            if hasattr(self.run_config, k):
                setattr(self.run_config, k, v)
        
        # Ensure device is set correctly in RunConfig
        self.run_config.device = self.device
        
        # Initialize IP-Adapter Instruct
        self.ip_instruct_model = self._init_ip_instruct_model(self.run_config)
        
        # Initialize BLIP2
        self._load_blip()
        
        # Load Inference Pipeline components for factory exposure
        # Note: The actual pipeline is assembled in __call__ or we can pre-assemble it here.
        # To expose components to factory, we should load them here.
        
        print(f"Loading SDXL components from {self.run_config.base_model_path}...")
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16
        ).to(self.device)
        
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(self.device)
        self.vae.to(dtype=torch.float16)
        print(f"VAE dtype: {self.vae.dtype}, force_upcast: {self.vae.config.force_upcast}")
        
        # We need to load the pipeline to get UNet and Text Encoders
        # We'll load a temporary pipeline to extract components or just load components directly.
        # Loading full pipeline is easier.
        
        # Load ControlNet (default to Canny for initialization, will be swapped in __call__ if needed)
        controlnet = ControlNetModel.from_pretrained(
            self.run_config.canny_controlnet_path, 
            torch_dtype=torch.float16, 
            variant="fp16"
        ).to(self.device)

        self.pipe_inference = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            self.run_config.base_model_path,
            controlnet=controlnet,
            vae=self.vae,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)
        
        self.pipe_inference.scheduler = UniPCMultistepScheduler.from_config(self.pipe_inference.scheduler.config)
        self.pipe_inference.unet.enable_gradient_checkpointing()
        # self.pipe_inference.enable_model_cpu_offload()
        
        # Load Standard IP-Adapter
        self.pipe_inference.load_ip_adapter(
            self.run_config.IP_path, subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
            image_encoder_folder=None,
        )
        scale_style = {
            "up": {"block_0": [0.0, 2.5, 0.0]},
        }
        self.pipe_inference.set_ip_adapter_scale(scale_style)
        
        # Expose components for factory
        self.unet = self.pipe_inference.unet
        self.text_encoder = self.pipe_inference.text_encoder
        self.text_encoder_2 = self.pipe_inference.text_encoder_2
        
        self.is_initialized = True
        print("StyleSSP initialized successfully.")

    def _load_blip(self):
        if self.blip_model is None:
            print("Loading BLIP2 model...")
            MODEL_ID = "Salesforce/blip2-flan-t5-xl"
            self.blip_processor = AutoProcessor.from_pretrained(MODEL_ID)
            self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
                MODEL_ID, device_map=self.device, load_in_8bit=False, torch_dtype=torch.float16
            )
            self.blip_model.eval()
            # Expose for factory
            self.blip2_model = self.blip_model

    def _init_ip_instruct_model(self, config):
        print("Initializing IP-Adapter Instruct...")
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
        if config.choose_pipeline == "sd15":
            ip_ckpt = config.ip_ckpt if hasattr(config, 'ip_ckpt') else "./checkpoints/models/ip-adapter-instruct-sd15.bin"
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            pipe = StableDiffusionPipelineCFG.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                scheduler=noise_scheduler,
                torch_dtype=torch.float16,
                feature_extractor=None,
                safety_checker=None,
            )
            ip_model = IPAdapterInstruct(
                sd_pipe=pipe, 
                image_encoder_path=image_encoder_path,
                ip_ckpt=ip_ckpt,
                device=config.device,
                dtypein=torch.float16,
                num_tokens=16
            )
        else:
            ip_ckpt = self.ip_instruct_ckpt if self.ip_instruct_ckpt else "./checkpoints/models/ip-adapter-instruct-sdxl.bin"
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            pipe = StableDiffusionXLPipelineExtraCFG.from_pretrained(
                config.base_model_path,
                scheduler=noise_scheduler,
                torch_dtype=torch.float16,
                feature_extractor=None,
            )
            ip_model = IPAdapterInstructSDXL(
                sd_pipe=pipe, 
                image_encoder_path=image_encoder_path,
                ip_ckpt=ip_ckpt,
                device=config.device,
                dtypein=torch.float16,
                num_tokens=16
            )
        return ip_model

    def generate_caption(self, image: Image.Image, text: str = None):
        self._load_blip()
        if text is not None:
            inputs = self.blip_processor(images=image, text=text, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.blip_model.generate(**inputs)
        else:
            inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.blip_model.generate(
                pixel_values=inputs.pixel_values,
                do_sample=True,
                temperature=1.0,
                length_penalty=1.0,
                repetition_penalty=1.5,
                max_length=50,
                min_length=1,
                num_beams=5,
                top_p=0.9,
            )
        result = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return result

    def get_depth_map(self, image):
        if self.depth_estimator is None:
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(self.device)
            self.depth_feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            
        image_tensor = self.depth_feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            depth_map = self.depth_estimator(image_tensor).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def get_canny_map(self, input_image_cv2):
        input_image_cv2 = cv2.Canny(input_image_cv2, 100, 200)
        input_image_cv2 = input_image_cv2[:, :, None]
        input_image_cv2 = np.concatenate([input_image_cv2, input_image_cv2, input_image_cv2], axis=2)
        anyline_image = Image.fromarray(input_image_cv2)
        return anyline_image

    def __call__(self, content: torch.Tensor, style: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """
        Run style transfer on tensor inputs.
        
        Args:
            content: Content image tensor [C, H, W] in range [0, 1]
            style: Style image tensor [C, H, W] in range [0, 1]
            alpha: Interpolation factor (not used in StyleSSP currently)
            
        Returns:
            Stylized image tensor [C, H, W] in range [0, 1]
        """
        config = self.run_config

        native_size = self.get_native_image_size()
        
        # Resize to native resolution if needed
        if content.shape[-2:] != (native_size, native_size):
            content = torch.nn.functional.interpolate(
                content, size=(native_size, native_size),
                mode='bilinear', align_corners=False
            )
        if style.shape[-2:] != (native_size, native_size):
            style = torch.nn.functional.interpolate(
                style, size=(native_size, native_size),
                mode='bilinear', align_corners=False
            )
        
        # Convert to PIL for BLIP2
        content_image = self._tensor_to_pil(content)
        style_image = self._tensor_to_pil(style)
        
        # # Handle batch dimension
        # if content.ndim == 4:
        #     content = content.squeeze(0)
        # if style.ndim == 4:
        #     style = style.squeeze(0)
        
        # # Convert tensors to PIL
        # to_pil = transforms.ToPILImage()
        # content_image = to_pil(content).convert("RGB").resize((config.resolution, config.resolution))
        # style_image = to_pil(style).convert("RGB").resize((config.resolution, config.resolution))

        # Generate captions
        # Note: In a real pipeline, we might want to cache these or allow passing them
        style_image_prompt = self.generate_caption(style_image)
        content_image_prompt = self.generate_caption(content_image)
        
        # print(f"Style Prompt: {style_image_prompt}")
        # print(f"Content Prompt: {content_image_prompt}")

        # Get embeddings
        content_instruct_prompt = "use the composition from the image"
        style_instruct_prompt = "use the style from the image"
        
        style_embeddings_instruct = self.ip_instruct_model.get_decouple_embeds(pil_image=style_image, prompt="", query=style_instruct_prompt)
        style_content_embeddings = self.ip_instruct_model.get_decouple_embeds(pil_image=style_image, prompt="", query=content_instruct_prompt)
        
        content_embeddings_instruct = self.ip_instruct_model.get_decouple_embeds(pil_image=content_image, prompt="", query=content_instruct_prompt)
        content_style_instruct = self.ip_instruct_model.get_decouple_embeds(pil_image=content_image, prompt="", query=style_instruct_prompt)

        # Inversion
        # We need to create temporary pipes for inversion as per original code
        pipe_inversion, pipe_inference_temp = get_pipes(Model_Type.SDXL, config.scheduler_type, device=self.device, model_name=config.base_model_path)
        
        with torch.enable_grad():
            _, inv_latent, _, _ = invert(
                content_image,
                content_image_prompt,
                config,
                pipe_inversion=pipe_inversion,
                pipe_inference=pipe_inference_temp,
                do_reconstruction=False,
                feature_extractor=self.ip_instruct_model,
                style_embedding=style_embeddings_instruct,
                content_embedding=content_embeddings_instruct,
                neg_style_embedding=content_style_instruct,
                neg_content_embedding=style_content_embeddings,
                enable_guidance=False,
                used_NPI_guidance=True
            )
        
        # Frequency manipulation
        latent_h, latent_l, latent_sum = freq_exp(inv_latent, d_s=0.3, d_t=0.9, alpha=0.7, filter_type="gaussian_b")
        latent_l = latent_l.to(inv_latent.dtype)
        
        # Cleanup inversion pipes
        del pipe_inversion, pipe_inference_temp
        torch.cuda.empty_cache()

        # Prepare ControlNet
        control_type = config.control_type
        cond_image = None
        controlnet_conditioning_scale = 1.0
        
        # Swap ControlNet in pipe_inference if needed
        # Note: self.pipe_inference was initialized with Canny. If control_type changes, we might need to reload.
        # For efficiency, we assume Canny for now or reload if different.
        
        if control_type == "tile":
            cond_image = content_image.resize((config.resolution, config.resolution))
            # Reload controlnet if needed (simplified for now, assuming we stick to one or reload)
            # In a full implementation, we'd manage multiple controlnets.
            # Here we just load the one we need into the pipe.
            controlnet = ControlNetModel.from_pretrained(config.tile_controlnet_path, torch_dtype=torch.float16, use_safetensors=True).to(self.device)
            self.pipe_inference.controlnet = controlnet
            controlnet_conditioning_scale = 0.25
            
        elif control_type == "canny":
            input_image_cv2 = np.array(content_image)
            anyline_image = self.get_canny_map(input_image_cv2)
            cond_image = anyline_image.resize((config.resolution, config.resolution))
            
            # Check if current controlnet is Canny (it is by default init)
            # If not, reload.
            # For safety, just reload or check config.
            controlnet = ControlNetModel.from_pretrained(config.canny_controlnet_path, torch_dtype=torch.float16, variant="fp16").to(self.device)
            self.pipe_inference.controlnet = controlnet
            controlnet_conditioning_scale = 0.2

        elif control_type == "tile_canny":
            cond_tile_image = content_image.resize((config.resolution, config.resolution))
            
            input_image_cv2 = np.array(content_image)
            anyline_image = self.get_canny_map(input_image_cv2)
            cond_canny_image = anyline_image.resize((config.resolution, config.resolution))
            
            controlnets = [
                ControlNetModel.from_pretrained(config.tile_controlnet_path, torch_dtype=torch.float16, use_safetensors=True).to(self.device),
                ControlNetModel.from_pretrained(config.canny_controlnet_path, torch_dtype=torch.float16, variant="fp16").to(self.device)
            ]
            self.pipe_inference.controlnet = MultiControlNetModel(controlnets)
            
            cond_image = [cond_tile_image, cond_canny_image]
            controlnet_conditioning_scale = [0.25, 0.40]
            
        elif control_type == "depth":
            depth_image = self.get_depth_map(content_image)
            cond_image = depth_image.resize((config.resolution, config.resolution))
            controlnet = ControlNetModel.from_pretrained(config.depth_controlnet_path, torch_dtype=torch.float16, variant="fp16").to(self.device)
            self.pipe_inference.controlnet = controlnet
            controlnet_conditioning_scale = 0.4

        # Generate
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        entire_mask = Image.new("RGB", (config.resolution, config.resolution), color=(255, 255, 255))

        # Ensure VAE is in float16
        self.pipe_inference.vae.to(dtype=torch.float16)

        output_pil = self.pipe_inference(
            prompt=content_image_prompt,
            negative_prompt="watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            num_inference_steps=config.num_inference_steps,
            eta=1.0,
            mask_image=entire_mask,
            image=content_image,
            control_image=cond_image,
            ip_adapter_image=style_image,
            generator=generator,
            latents=latent_l,
            guidance_scale=config.guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            npi_interp=0.5,
            style_embeddings_instruct=style_embeddings_instruct,
            content_embeddings_instruct=content_embeddings_instruct,
            style_guidance_scale=config.style_guidance_scale,
            content_guidance_scale=config.content_guidance_scale,
            ip_instruct_model=self.ip_instruct_model,
            CSD_model=None,
            inv_guidance=config.inv_guidance,
            feature_extractor=self.ip_instruct_model,
        ).images[0]

        # Convert back to tensor
        output_tensor = self._pil_to_tensor(output_pil)
        
        return output_tensor

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor [1, 3, H, W] to PIL Image."""
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = (image * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)
    
    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor [1, 3, H, W]."""
        image = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor