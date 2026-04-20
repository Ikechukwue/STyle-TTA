"""
xAILab Bamberg
University of Bamberg

@description:
PhotoNAS method implementation (inference only, using pretrained checkpoint).
Based on: https://github.com/pkuanjie/StyleNAS/tree/master/PhotoNAS

Pretrained checkpoint available at:
https://drive.google.com/open?id=15PP0K55jH2tBeWfLAYG7r0LuW0RZvmKd
"""

from typing import Optional, Dict
import os
import torch

from experiments.reference_methods.style_transfer.photorealistic.photonas.net import (
    PhotoNASEncoder, PhotoNASDecoder
)
from experiments.reference_methods.style_transfer.photorealistic.photonas.wct import transform


class Method:
    """
    PhotoNAS (Ultrafast Photorealistic Style Transfer via Neural Architecture Search) method.
    
    Paper: "Ultrafast Photorealistic Style Transfer via Neural Architecture Search"
    Authors: Jie An, Haoyi Xiong, Jun Huan, Jiebo Luo
    Conference: AAAI 2020
    
    This is a training-free implementation that uses the official pretrained checkpoint.
    """
    
    def __init__(self, pretrained_weights: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize PhotoNAS method.
        
        Args:
            pretrained_weights: Path to pretrained decoder checkpoint (photonas.pth.tar)
            device: Device to use for inference
        """
        self.device = device
        self.pretrained_weights = pretrained_weights
        self.encoder = None
        self.decoder = None
        self.is_initialized = False
        self.config = self.get_default_config()
        
        if pretrained_weights and os.path.exists(pretrained_weights):
            self._initialize_network(pretrained_weights)
        else:
            print(f"Warning: No weights provided for PhotoNAS or file not found at {pretrained_weights}. Method will fail if run.")
            
    def get_default_config(self) -> Dict:
        """
        Get default configuration.
        """
        return {
            'native_image_size': 512,  # Official code uses 512 for inference
            'method_name': 'photonas',
            'd_control': '01010000000100000000000000001111'  # Default architecture from official repo
        }
    
    @staticmethod
    def get_native_image_size() -> int:
        """Return the native image size for this method."""
        return 512
        
    def _initialize_network(self, path: str):
        """Load model weights from pretrained checkpoint."""
        print(f"Loading PhotoNAS checkpoint from {path}")
        
        # Initialize encoder with pretrained VGG weights
        self.encoder = PhotoNASEncoder(pretrained=True)
        self.decoder = PhotoNASDecoder()
        
        # Load decoder weights from checkpoint
        # Official checkpoint is decoder-only (decoder_epoch_2.pth.tar)
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'encoder' in state_dict and 'decoder' in state_dict:
            # Full model checkpoint
            self.decoder.load_state_dict(state_dict['decoder'])
        else:
            # Decoder-only checkpoint (official format)
            self.decoder.load_state_dict(state_dict)
                
        # Move to device and set eval mode
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        
        self.is_initialized = True
        print("PhotoNAS model initialized successfully")

    def _parse_control(self, d_control: str):
        """Parse the d_control string into control lists for each decoder stage."""
        d0 = [int(x) for x in d_control[:5]]
        d1 = [int(x) for x in d_control[5:8]]
        d2 = [int(x) for x in d_control[9:16]]
        d3 = [int(x) for x in d_control[16:23]]
        d4 = [int(x) for x in d_control[23:28]]
        d5 = [int(x) for x in d_control[28:32]]
        return d0, d1, d2, d3, d4, d5

    def __call__(self, content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Perform style transfer.
        
        Args:
            content: Content image tensor [C, H, W] or [B, C, H, W] in [0, 1]
            style: Style image tensor [C, H, W] or [B, C, H, W] in [0, 1]
            alpha: Style strength (0-1, default 1.0)
            
        Returns:
            Stylized image tensor [C, H, W] or [B, C, H, W] in [0, 1]
        """
        if not self.is_initialized:
            raise RuntimeError("PhotoNAS model not initialized (missing weights).")
        
        # Handle single image input (C, H, W) -> (1, C, H, W)
        squeeze_output = False
        if content.dim() == 3:
            content = content.unsqueeze(0)
            squeeze_output = True
        if style.dim() == 3:
            style = style.unsqueeze(0)
            
        # Move to device
        content = content.to(self.device)
        style = style.to(self.device)
        
        d_control = self.config['d_control']
        d0, d1, d2, d3, d4, d5 = self._parse_control(d_control)
        
        with torch.no_grad():
            # 1. Encode content and style
            # Returns: relu5_1, relu4_1, relu3_1, relu2_1, relu1_1
            cF = list(self.encoder(content))
            sF = list(self.encoder(style))
            
            # 2. Apply WCT on initial encoder features based on d_control
            # d0[0]: WCT on bottleneck (relu5_1)
            # d2[-1], d3[-1], d4[-1], d5[-1]: WCT on skip connections
            csF = []
            for i in range(len(cF)):
                apply_wct = False
                if i == 0 and d0[0] == 1: apply_wct = True
                elif i == 1 and d2[-1] == 1: apply_wct = True
                elif i == 2 and d3[-1] == 1: apply_wct = True
                elif i == 3 and d4[-1] == 1: apply_wct = True
                elif i == 4 and d5[-1] == 1: apply_wct = True
                
                if apply_wct:
                    csF.append(transform(cF[i], sF[i], alpha))
                else:
                    csF.append(cF[i])
            
            # 3. Decode stage-by-stage with intermediate WCT
            # Stage 0: Pyramid feature fusion
            curr_feat = self.decoder.forward_stage0(csF[0], csF[1:], d_control)
            curr_style_feat = self.decoder.forward_stage0(sF[0], sF[1:], d_control)
            
            # WCT after Stage 0? -> d1[0]
            if d1[0] == 1:
                curr_feat = transform(curr_feat, curr_style_feat, alpha)
                
            # Stage 1
            curr_feat = self.decoder.forward_stage1(curr_feat, csF[1:], d_control)
            curr_style_feat = self.decoder.forward_stage1(curr_style_feat, sF[1:], d_control)
            
            # WCT after Stage 1? -> d2[0]
            if d2[0] == 1:
                curr_feat = transform(curr_feat, curr_style_feat, alpha)
                
            # Stage 2
            curr_feat = self.decoder.forward_stage2(curr_feat, csF[1:], d_control)
            curr_style_feat = self.decoder.forward_stage2(curr_style_feat, sF[1:], d_control)
            
            # WCT after Stage 2? -> d3[0]
            if d3[0] == 1:
                curr_feat = transform(curr_feat, curr_style_feat, alpha)
                
            # Stage 3
            curr_feat = self.decoder.forward_stage3(curr_feat, csF[1:], d_control)
            curr_style_feat = self.decoder.forward_stage3(curr_style_feat, sF[1:], d_control)
            
            # WCT after Stage 3? -> d4[0]
            if d4[0] == 1:
                curr_feat = transform(curr_feat, curr_style_feat, alpha)
                
            # Stage 4
            curr_feat = self.decoder.forward_stage4(curr_feat, csF[1:], d_control)
            curr_style_feat = self.decoder.forward_stage4(curr_style_feat, sF[1:], d_control)
            
            # WCT after Stage 4? -> d5[0]
            if d5[0] == 1:
                curr_feat = transform(curr_feat, curr_style_feat, alpha)
                
            # Stage 5 (Final output)
            out = self.decoder.forward_stage5(curr_feat, csF[1:], d_control)
            
            out = out.clamp(0, 1)
            
            if squeeze_output:
                out = out.squeeze(0)
                
            return out
