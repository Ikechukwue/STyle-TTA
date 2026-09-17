# modified from https://github.com/unity-research/IP-Adapter-Instruct/blob/main/ip_adapter/ip_adapter.py
# Copyright 2024 unity-research/IP-Adapter-Instruct
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List
import torch.nn as nn
import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, AutoImageProcessor, AutoModel, CLIPTokenizer, CLIPTextModel
from code.experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.utils import is_torch2_available, get_generator
from code.experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.ip_joint_attention import IPJointAttnProcessor2_0, JointAttnProcessor2_0
from code.experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.joint_attention_block_modified import JointTransformerBlock_IP
from code.experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.resampler_Instruct import ResamplerInstruct, MLP, ResamplerInstructBigger
from code.experiments.reference_methods.style_transfer.artistic_diffusion.stylessp.ip_adapter.ip_adapter import IPAdapter
# from .resampler_SD3 import ResamplerSD3, ResamplerSD3_Instruct

class IPAdapterInstruct(IPAdapter):
    """IP-Adapter with fine-grained features"""
    def __init__ (self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, dtypein=torch.float32):
        #super().__init__()
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.pipe = sd_pipe.to(self.device)
        if ip_ckpt is not None:
            self.set_ip_adapter()
        
        # load image encoder
        #self.image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.device, dtype=torch.float16)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=dtypein)
        #self.clip_image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()
        # print(ip_ckpt)
        if ip_ckpt is not None:
            self.load_ip_adapter()

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}, "mlp_proj": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        
        # print(state_dict["image_proj"].keys(),state_dict["ip_adapter"].keys())
        print(self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False))
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        print(ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False))

    def init_proj(self):
        image_proj_model = ResamplerInstructBigger(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim_image_embeds=self.image_encoder.config.hidden_size,
            embedding_dim_instruct_embeds=self.pipe.text_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
            ff_mult_secondary=2,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, instruct_embeds=None, negative_instruct_embeds=None, prompt_embeds=None, negative_prompt_embeds=None, instruct_embeds_everything=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        
        # print(clip_image_embeds.dtype,instruct_embeds.dtype)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds, instruct_embeds, prompt_embeds)
        
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds, negative_instruct_embeds, negative_prompt_embeds)
        
        img_only_prompt_embeds = self.image_proj_model(clip_image_embeds, negative_prompt_embeds, negative_prompt_embeds)
        img_prompt_everything_cond = self.image_proj_model(clip_image_embeds, instruct_embeds_everything, prompt_embeds)
        
        return image_prompt_embeds, uncond_image_prompt_embeds, img_only_prompt_embeds, img_prompt_everything_cond

    # @torch.inference_mode()
    @torch.enable_grad()
    def get_single_image_embeds(self, pil_image=None, clip_image=None, clip_image_embeds=None, instruct_embeds=None, prompt_embeds=None):
        if clip_image is None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
        
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        # print(clip_image_embeds.dtype,instruct_embeds.dtype)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds, instruct_embeds, prompt_embeds)
        return image_prompt_embeds

    @torch.enable_grad()
    def get_decouple_embeds(
        self,
        pil_image=None,
        clip_image=None,
        clip_image_embeds=None,
        prompt=None,
        query=None,
        prompt_embeds=None,
        query_embeds=None,
        **kwargs,
    ):
        if prompt is None:
            prompt = "best quality, high quality"
        
        # with torch.inference_mode():
        if prompt_embeds is None:
            prompt_ids_prompt = self.pipe.tokenizer(
                prompt, max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            small_prompt_embeds = self.pipe.text_encoder(prompt_ids_prompt, return_dict=True)[0]
        else:
            small_prompt_embeds = prompt_embeds
            
        if query_embeds is None:
            prompt_ids_instruction = self.pipe.tokenizer(
                query, max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            instruct_embeds = self.pipe.text_encoder(prompt_ids_instruction, return_dict=True)[0]
        else:
            instruct_embeds = query_embeds
            
        image_prompt_embeds = self.get_single_image_embeds(
            pil_image=pil_image, 
            clip_image=clip_image,
            clip_image_embeds=clip_image_embeds,
            instruct_embeds=instruct_embeds,
            prompt_embeds=small_prompt_embeds,
        )
        return image_prompt_embeds

    def generate(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        query=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=6.0,
        image_guidance_scale=1.0,
        instruct_guidance_scale=6.0,
        num_inference_steps=30,
        width=512,
        height=512,
        auto_scale=True,
        **kwargs,
    ):
        self.set_scale(scale)
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)
        
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        print("prompt = ", prompt)
        
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            
            prompt_ids_instruction = self.pipe.tokenizer(
                query, max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            instruct_embeds = self.pipe.text_encoder(prompt_ids_instruction, return_dict=True)[0]
            
            prompt_ids_instruction_negative = self.pipe.tokenizer(
                "", max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            
            #TEMPORARY UNTIL IVE TRAINED IT TO CONDITION ON IMAGE ONLY WITH NO INSTRUCTION
            prompt_ids_instruction_everything_temp = self.pipe.tokenizer(
                "everything", max_length=self.pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)
            instruct_embeds_everything = self.pipe.text_encoder(prompt_ids_instruction_everything_temp, return_dict=True)[0]
            
            negative_instruct_embeds = self.pipe.text_encoder(prompt_ids_instruction_negative, return_dict=True)[0]  
            
            prompt_embeds_ = prompt_embeds_[0].unsqueeze(0)
            negative_prompt_embeds_ = negative_prompt_embeds_[0].unsqueeze(0)
            
            image_prompt_embeds, uncond_image_prompt_embeds, img_only_prompt_embeds, img_prompt_everything_cond = self.get_image_embeds(
                pil_image=pil_image,
                clip_image_embeds=clip_image_embeds,
                instruct_embeds=instruct_embeds,
                negative_instruct_embeds=negative_instruct_embeds,
                prompt_embeds=prompt_embeds_,
                negative_prompt_embeds=negative_prompt_embeds_,
                instruct_embeds_everything=instruct_embeds_everything
            )
            
            prompt_embeds_ = prompt_embeds_.repeat(num_samples, 1, 1)
            negative_prompt_embeds_ = negative_prompt_embeds_.repeat(num_samples, 1, 1)
            
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            
            img_only_prompt_embeds = img_only_prompt_embeds.repeat(1, num_samples, 1)
            img_only_prompt_embeds = img_only_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            
            img_prompt_everything_cond = img_prompt_everything_cond.repeat(1, num_samples, 1)
            img_prompt_everything_cond = img_prompt_everything_cond.view(bs_embed * num_samples, seq_len, -1)
            
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
            img_only_prompt_embeds = torch.cat([negative_prompt_embeds_, img_only_prompt_embeds], dim=1)
            img_prompt_everything_cond = torch.cat([negative_prompt_embeds_, img_prompt_everything_cond], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        
        simple_cfg_mode = False
        if "style" in query or "colour" in query or "everything" in query or "color" in query or "face" in query or "facial" in query or "colour" in query:
            simple_cfg_mode = True
            
        #replace colour with color
        if "colour" in query:
            query = query.replace("colour","color")
            #llm disrespect generating the training queries ngl
            
        if auto_scale:
            if "composition" in query or "pose" in query:
                scale = scale - 0.2
            #else:
            #    scale = 0.9
        self.set_scale(scale)
        
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            img_only_prompt_embeds=img_only_prompt_embeds,
            img_prompt_everything_cond=img_prompt_everything_cond,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            instruct_guidance_scale=instruct_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            simple_cfg_mode=simple_cfg_mode,
            **kwargs,
        ).images
        
        return images

def replace_transformer_blocks(original_model):
    original_model.transformer_blocks = nn.ModuleList(
        [
            JointTransformerBlock_IP(
                dim=original_model.inner_dim,
                num_attention_heads=original_model.config.num_attention_heads,
                attention_head_dim=original_model.inner_dim,
                context_pre_only=i == original_model.config.num_layers - 1,
            )
            for i in range(original_model.config.num_layers)
        ]
    )
    return original_model

# class IPAdapter_sd3:
#     ... (omitted as not needed for SDXL)

# class IPAdapter_sd3_Instruct (IPAdapter_sd3):
#     ... (omitted as not needed for SDXL)

class IPAdapterInstructSDXL(IPAdapterInstruct):
    def init_proj(self):
        image_proj_model = ResamplerInstructBigger(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim_image_embeds=self.image_encoder.config.hidden_size,
            embedding_dim_instruct_embeds=self.pipe.text_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
            ff_mult_secondary=2,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
