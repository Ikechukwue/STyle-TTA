"""
xAILab Bamberg
University of Bamberg

@description:
ModFlows method implementation for reference methods framework (inference only).
Based on official PyTorch implementation: https://github.com/maria-larchenko/modflows

Pretrained weights download:
Model: https://huggingface.co/MariaLarchenko/modflows_color_encoder/tree/main (modflows_color_encoder_B6_dim_8195_iter_751001.pt)
"""

from typing import Optional, Dict, Union
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Import from local src
from code.experiments.reference_methods.style_transfer.photorealistic.modflows.src.encoder import Encoder
from code.experiments.reference_methods.style_transfer.photorealistic.modflows.src.inference import run_inference

class Method:
    """
    ModFlows (Color Transfer with Modulated Flows) - Training-free color transfer.
    
    Paper: "Color Transfer with Modulated Flows" (AAAI 2025)
    Authors: Maria Larchenko, Alexander Lobashev, Dmitry Guskov, Vladimir Vladimirovich Palyulin
    
    This method uses a pretrained Encoder to predict parameters for Neural ODEs (Rectified Flows)
    that map the color distribution of the content image to the style image.
    """
    
    def __init__(self, pretrained_weights: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self.pretrained_weights = pretrained_weights
        
        # 1. Initialize configs
        self.config = self.get_default_config()
        
        # 2. Load Model
        self.encoder = None
        
        if pretrained_weights:
            self._initialize_network(pretrained_weights)
        else:
            print("Warning: No weights provided for ModFlows. Method will fail if run.")

    def _initialize_network(self, weights_path: str):
        # The official code uses specific params for the Encoder:
        # k_dim=8195, input_dim=4, hidden=1024, output_dim=3
        self.encoder = Encoder(k_dim=8195, input_dim=4, hidden=1024, output_dim=3, device=self.device)
        
        print(f"Loading ModFlows weights from: {weights_path}")
        try:
            enc_params = torch.load(weights_path, map_location=self.device)
            self.encoder.load_state_dict(enc_params)
        except Exception as e:
            print(f"Error loading ModFlows weights: {e}")
            raise e
            
        self.encoder.eval()

    def get_default_config(self) -> Dict:
        return {
            'strength': 1.0,
            'steps': 8,  # Default from run_inference.py
            'native_image_size': 512, # Flexible, but 512 is a good default
            'method_name': 'modflows'
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 512

    def __call__(self, content: torch.Tensor, style: torch.Tensor, alpha: float = None) -> torch.Tensor:
        """
        Apply color transfer.
        
        Args:
            content: Content image tensor [C, H, W] in [0, 1]
            style: Style image tensor [C, H, W] in [0, 1]
            alpha: Strength of the effect (0.0 to 1.0). If None, uses default config.
            
        Returns:
            Stylized image tensor [C, H, W] in [0, 1]
        """
        # 1. Preprocess inputs
        # Convert tensors to PIL Images for the inference pipeline
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        
        # Handle batch dimension if present
        if content.dim() == 4:
            content = content.squeeze(0)
        if style.dim() == 4:
            style = style.squeeze(0)
            
        content_pil = to_pil(content.cpu())
        style_pil = to_pil(style.cpu())
        
        strength = alpha if alpha is not None else self.config['strength']
        steps = self.config['steps']
        
        # 2. Run Inference
        # run_inference returns: content_im, latent_im, styled_im, style_im
        _, _, styled_pil, _ = run_inference(
            encoder=self.encoder,
            device=self.device,
            content_im_path=content_pil,
            style_im_path=style_pil,
            compress=False, # We handle resizing outside if needed, or let it be native
            enc_steps=steps,
            strength=strength,
            crop=False
        )
        
        # 3. Postprocess
        styled_tensor = to_tensor(styled_pil).to(self.device)
        
        # Add batch dimension if needed (though usually expected [C,H,W] by caller, 
        # but some pipelines expect [1,C,H,W])
        # The interface says return [C, H, W], so we return that.
        
        return styled_tensor
