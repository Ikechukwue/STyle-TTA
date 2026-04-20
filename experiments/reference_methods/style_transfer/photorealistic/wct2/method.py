"""
xAILab Bamberg
University of Bamberg

@description:
WCT2 method implementation for reference methods framework (inference only).
Based on official PyTorch implementation: https://github.com/clovaai/WCT2

Pretrained weights download:
Model: https://github.com/clovaai/WCT2/tree/master/model_checkpoints (wave_encoder_cat5_l4.pt, wave_decoder_cat5_l4.pt)
"""

from typing import Optional, Dict, Union
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Import from local src
from experiments.reference_methods.style_transfer.photorealistic.wct2.src.transfer import WCT2
from experiments.reference_methods.style_transfer.photorealistic.wct2.src.utils.io import load_segment, compute_label_info, change_seg


def open_image(image_path, image_size=None):
    """Open and preprocess image with center crop for 16-divisibility."""
    image = Image.open(image_path).convert('RGB')
    _transforms = []
    if image_size is not None:
        image = transforms.Resize(image_size)(image)
    w, h = image.size
    _transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
    _transforms.append(transforms.ToTensor())
    transform = transforms.Compose(_transforms)
    return transform(image).unsqueeze(0)


def preprocess_image(image, image_size=None):
    """Preprocess PIL Image with center crop for 16-divisibility."""
    if image_size is not None:
        image = transforms.Resize(image_size)(image)
    w, h = image.size
    _transforms = []
    _transforms.append(transforms.CenterCrop((h // 16 * 16, w // 16 * 16)))
    _transforms.append(transforms.ToTensor())
    transform = transforms.Compose(_transforms)
    return transform(image).unsqueeze(0)


def preprocess_segment(segment, image_size=None):
    """Preprocess segmentation mask with center crop for 16-divisibility."""
    if segment is None or (isinstance(segment, np.ndarray) and segment.size == 0):
        return np.asarray([])
    if isinstance(segment, str):
        segment = Image.open(segment)
    if image_size is not None:
        transform = transforms.Resize(image_size, interpolation=Image.NEAREST)
        segment = transform(segment)
    w, h = segment.size
    transform = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))
    segment = transform(segment)
    if len(np.asarray(segment).shape) == 3:
        segment = change_seg(segment)
    return np.asarray(segment)

class Method:
    """
    WCT2 (Photorealistic Style Transfer via Wavelet Transforms) - Training-free color/style transfer.
    
    Paper: "Photorealistic Style Transfer via Wavelet Transforms" (ICCV 2019)
    Authors: Jaejun Yoo, Youngjung Uh, Sanghyuk Chun, Byeongkyu Kang, Jung-Woo Ha
    
    This method uses a Wavelet-based Encoder-Decoder architecture with Whitening and Coloring Transforms (WCT)
    applied at multiple levels (Encoder, Decoder, Skip connections).
    """
    
    def __init__(self, pretrained_weights: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self.pretrained_weights = pretrained_weights
        self.wct2 = None
        self.config = self.get_default_config()
        
        if pretrained_weights:
            self._initialize_network(pretrained_weights)
        else:
            print("Warning: No pretrained_weights provided for WCT2. Method will fail if run.")

    def _initialize_network(self, weights_path: str):
        weights = {}
        if os.path.isfile(weights_path):
            try:
                print(f"Loading WCT2 weights from: {weights_path}")
                weights = torch.load(weights_path, map_location=self.device)
            except Exception as e:
                print(f"Error loading WCT2 weights: {e}")
        else:
            print(f"Warning: pretrained_weights {weights_path} is not a file. WCT2 requires a single .pt file containing encoder and decoder weights.")
        
        # Initialize WCT2
        # We use the default configuration: transfer_at=['encoder', 'decoder', 'skip'], option_unpool='cat5'
        try:
            self.wct2 = WCT2(weights, transfer_at=['encoder', 'decoder', 'skip'], option_unpool='cat5', device=self.device)
        except Exception as e:
            print(f"Error initializing WCT2: {e}")
            self.wct2 = None

    def get_default_config(self) -> Dict:
        return {
            'alpha': 1.0,
            'native_image_size': 512,
            'method_name': 'wct2'
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 512

    def __call__(self, content: torch.Tensor, style: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass for color transfer.
        
        Args:
            content: Content image tensor [B, C, H, W]
            style: Style image tensor [B, C, H, W]
            kwargs:
                alpha (float): Interpolation factor (0.0 - 1.0). Default 1.0.
                content_segment (Tensor): Segmentation mask for content.
                style_segment (Tensor): Segmentation mask for style.
        
        Returns:
            Tensor: Result image [B, C, H, W]
        """
        if self.wct2 is None:
            raise RuntimeError("WCT2 model not initialized properly.")
            
        alpha = kwargs.get('alpha', 1.0)
        content_segment = kwargs.get('content_segment', None)
        style_segment = kwargs.get('style_segment', None)
        
        # WCT2 implementation expects [1, C, H, W]
        return self.wct2.transfer(content, style, content_segment, style_segment, alpha=alpha)

    def transfer_color(self, source_image: Union[str, Image.Image], target_image: Union[str, Image.Image], **kwargs) -> Image.Image:
        """
        Transfers color/style from target_image to source_image using WCT2.
        
        Args:
            source_image: Content image (path or PIL Image)
            target_image: Style image (path or PIL Image)
            kwargs:
                alpha (float): Interpolation factor (0.0 - 1.0). Default 1.0.
                image_size (int): Optional resize before processing. Default None.
                content_segment (str, PIL Image or np.ndarray): Segmentation mask for content.
                style_segment (str, PIL Image or np.ndarray): Segmentation mask for style.
        
        Returns:
            PIL Image: Result image
        """
        if self.wct2 is None:
            raise RuntimeError("WCT2 model not initialized properly (likely missing weights).")

        alpha = kwargs.get('alpha', 1.0)
        image_size = kwargs.get('image_size', None)
        content_segment_input = kwargs.get('content_segment', None)
        style_segment_input = kwargs.get('style_segment', None)

        # 1. Prepare content and style images with proper preprocessing (16-divisibility)
        if isinstance(source_image, str):
            content = open_image(source_image, image_size).to(self.device)
        else:
            content = preprocess_image(source_image, image_size).to(self.device)
            
        if isinstance(target_image, str):
            style = open_image(target_image, image_size).to(self.device)
        else:
            style = preprocess_image(target_image, image_size).to(self.device)

        # 2. Prepare segmentation masks with proper preprocessing
        content_segment = preprocess_segment(content_segment_input, image_size)
        style_segment = preprocess_segment(style_segment_input, image_size)

        # 3. Run Transfer
        with torch.no_grad():
            out = self(content, style, content_segment=content_segment, style_segment=style_segment, alpha=alpha)
        
        # 4. Post-process
        out = out.squeeze(0).cpu().clamp(0, 1)
        out_img = transforms.ToPILImage()(out)
        
        return out_img
