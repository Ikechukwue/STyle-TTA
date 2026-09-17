"""
xAILab Bamberg
University of Bamberg

@description:
DeepPreset method implementation for reference methods framework (inference only).
Based on official PyTorch implementation: https://github.com/minhmanho/deep_preset

Pretrained weights download:
Model: https://drive.google.com/file/d/1GegyHf3OD17k_WID3-vA7S8nRQwPfpTC/view (dp_wPPL.tar)
"""

from typing import Optional, Dict
import torch
import os
from types import SimpleNamespace
from code.experiments.reference_methods.style_transfer.photorealistic.deeppreset.src.dp import DeepPreset


class Method:
    def __init__(self, pretrained_weights: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self.pretrained_weights = pretrained_weights
        self.model = None
        
        self.config = self.get_default_config()
        
        if pretrained_weights and os.path.exists(pretrained_weights):
            self._initialize_network(pretrained_weights)
        else:
            print(f"Warning: No weights provided for DeepPreset or file not found at {pretrained_weights}. Method will fail if run.")

    def _initialize_network(self, weights_path: str):
        args = SimpleNamespace(
            ckpt=weights_path,
            device=self.device,
            size="512x512",
            p_only=False
        )
        self.model = DeepPreset(args)

    def get_default_config(self) -> Dict:
        return {
            'native_image_size': 512, 
            'method_name': 'deeppreset'
        }

    @staticmethod
    def get_native_image_size() -> int:
        return 512

    def __call__(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("DeepPreset model not initialized (missing weights).")
            
        # content, style: [C, H, W] in [0, 1]
        # DeepPreset.process handles normalization and batch dim
        
        out = self.model.process(content, style)
        
        # out is [1, C, H, W]
        return out.squeeze(0)
