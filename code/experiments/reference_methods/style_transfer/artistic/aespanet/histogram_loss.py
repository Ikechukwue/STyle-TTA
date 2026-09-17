"""
xAILab Bamberg
University of Bamberg

@description:
RGBuv Histogram Loss implementation for AesPA-Net
Based on: https://github.com/Kibeom-Hong/AesPA-Net/blob/main/hist_loss.py

This implements the official differentiable histogram loss used in AesPA-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def rgb_to_uv_log(rgb):
    """
    Convert RGB to UV space using log-ratio (official AesPA-Net implementation).
    
    This matches the official hist_loss.py implementation:
    Iu = log(R/G), Iv = log(R/B)
    
    Args:
        rgb: RGB image tensor (B, 3, H, W) in range [0, 1]
        
    Returns:
        UV image tensor (B, 2, H, W) in range approximately [-3, 3]
    """
    EPS = 1e-6
    
    # Extract RGB channels
    R = rgb[:, 0:1, :, :]
    G = rgb[:, 1:2, :, :]
    B = rgb[:, 2:3, :, :]
    
    # Compute log ratios: u = log(R/G), v = log(R/B)
    Iu = torch.log(R + EPS) - torch.log(G + EPS)
    Iv = torch.log(R + EPS) - torch.log(B + EPS)
    
    return torch.cat([Iu, Iv], dim=1)


class HistogramLayer(nn.Module):
    """
    Differentiable histogram layer using soft binning.
    
    Based on official implementation: hist_loss.py:25-91
    """
    
    def __init__(self, nbins=256, method='inverse-quadratic', sigma=0.02):
        """
        Initialize histogram layer.
        
        Args:
            nbins: Number of histogram bins (default: 256 to match official)
            method: Kernel method ('inverse-quadratic' or 'thresholding')
            sigma: Bandwidth parameter for soft binning
        """
        super(HistogramLayer, self).__init__()
        self.nbins = nbins
        self.method = method
        self.sigma = sigma
        
        # Bin centers uniformly distributed in [-3, 3]
        # This covers ~99.7% of log-transformed data
        self.register_buffer('centers', torch.linspace(-3, 3, nbins))
    
    def forward(self, x):
        """
        Compute differentiable histogram.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Histogram tensor (B, C*nbins)
        """
        B, C, H, W = x.size()
        
        # Reshape to (B*C, H*W) for vectorized computation
        x_flat = x.view(B * C, -1)  # (B*C, H*W)
        
        # Expand dimensions for broadcasting
        # x_flat: (B*C, H*W, 1)
        # centers: (1, 1, nbins)
        x_expanded = x_flat.unsqueeze(-1)  # (B*C, H*W, 1)
        centers_expanded = self.centers.view(1, 1, -1)  # (1, 1, nbins)
        
        # Compute absolute distance from each pixel to each bin center
        # diff: (B*C, H*W, nbins)
        diff = torch.abs(x_expanded - centers_expanded)
        
        # Soft binning using inverse-quadratic kernel (matches official)
        if self.method == 'inverse-quadratic':
            # w(x) = 1 / (d^2 / sigma^2 + 1)
            diff_normalized = (diff ** 2) / (self.sigma ** 2)
            weight = 1.0 / (diff_normalized + 1.0)
        elif self.method == 'thresholding':
            # Thresholding method
            eps = 6.0 / self.nbins
            weight = (diff <= eps / 2).float()
        elif self.method == 'RBF':
            # RBF kernel
            diff_normalized = (diff ** 2) / (self.sigma ** 2)
            weight = torch.exp(-diff_normalized)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Sum weights across spatial dimensions: (B*C, H*W, nbins) -> (B*C, nbins)
        hist = weight.sum(dim=1)
        
        # Normalize to get probability distribution
        hist_normalized = hist / (hist.sum(dim=1, keepdim=True) + 1e-6)
        
        # Reshape to (B, C*nbins)
        hist_flat = hist_normalized.view(B, C * self.nbins)
        
        return hist_flat


class RGBuvHistBlock(nn.Module):
    """
    RGB-uv histogram block for color histogram loss.
    
    This computes 5-channel histograms:
    - 3 channels from RGB
    - 2 channels from UV (log-based color space)
    
    Based on official implementation: hist_loss.py:94-176
    CRITICAL: Official uses h=256 (number of bins), insz=64 (resize dimension)
    """
    
    def __init__(self, h=256, insz=64, resizing='interpolation', 
                 method='inverse-quadratic', sigma=0.02, intensity_scale=True):
        """
        Initialize RGBuv histogram block.
        
        Args:
            h: Number of histogram bins (default: 256 - MATCHES OFFICIAL)
            insz: Input size for resizing (default: 64 - MATCHES OFFICIAL)
            resizing: Resizing method ('interpolation' or 'sampling')
            method: Histogram kernel method
            sigma: Bandwidth for soft binning
            intensity_scale: Whether to scale intensities to [-3, 3]
        """
        super(RGBuvHistBlock, self).__init__()
        self.h = h
        self.insz = insz
        self.resizing = resizing
        self.intensity_scale = intensity_scale
        
        # Create histogram layer
        self.hist = HistogramLayer(nbins=h, method=method, sigma=sigma)
    
    def forward(self, x):
        """
        Compute RGB-uv histogram.
        
        Args:
            x: Input image (B, 3, H, W) in range [0, 1]
            
        Returns:
            Histogram features (B, 5*h) where h is number of bins
        """
        # CRITICAL: Clamp to [0, 1] range (official does this)
        x = torch.clamp(x, 0, 1)
        
        B, C, H, W = x.size()
        
        # Resize input if needed (for efficiency)
        if self.resizing == 'interpolation' and (H > self.insz or W > self.insz):
            x_resized = F.interpolate(x, size=(self.insz, self.insz), 
                                     mode='bilinear', align_corners=False)
        else:
            x_resized = x
        
        # 1. Compute RGB histograms (3 channels)
        # Scale RGB to [-3, 3] range if intensity_scale is True
        if self.intensity_scale:
            # Official: Iy = (I - 0.5) / 0.5 (maps [0,1] to [-1,1], then implicitly scaled)
            # For histogram in [-3, 3], we map [0, 1] to approximately [-3, 3]
            rgb_scaled = (x_resized - 0.5) * 6.0  # Maps [0,1] to [-3, 3]
        else:
            rgb_scaled = x_resized
        
        hist_rgb = self.hist(rgb_scaled)  # (B, 3*h)
        
        # 2. Convert to UV space using log ratios (official implementation)
        x_uv = rgb_to_uv_log(x_resized)  # (B, 2, H, W)
        
        # UV channels are already in approximately [-3, 3] range from log transform
        # No additional scaling needed
        
        # 3. Compute u,v histograms (2 channels)
        hist_uv = self.hist(x_uv)  # (B, 2*h)
        
        # 4. Concatenate all histograms: RGB (3*h) + uv (2*h) = 5*h
        hist = torch.cat([hist_rgb, hist_uv], dim=1)  # (B, 5*h)
        
        return hist


def histogram_loss(input_hist, target_hist, eps=1e-6):
    """
    Compute histogram loss using Hellinger distance (official implementation).
    
    Official formula from baseline.py:
    loss = (1/sqrt(2)) * sqrt(sum((sqrt(target) - sqrt(input))^2)) / batch_size
    
    This is the Hellinger distance between histogram distributions.
    
    Args:
        input_hist: Input histogram (B, 5*h)
        target_hist: Target histogram (B, 5*h)
        eps: Small value for numerical stability
        
    Returns:
        Histogram loss scalar
    """
    # Compute Hellinger distance (official implementation)
    # 1/sqrt(2) * sqrt(sum((sqrt(target_hist) - sqrt(input_hist))^2)) / batch_size
    loss = (1.0 / np.sqrt(2.0)) * (
        torch.sqrt(
            torch.sum(
                torch.pow(
                    torch.sqrt(target_hist + eps) - torch.sqrt(input_hist + eps), 
                    2
                )
            )
        ) / input_hist.shape[0]
    )
    return loss


class RGBuvHistogramLoss(nn.Module):
    """
    Complete RGBuv histogram loss module for AesPA-Net.
    
    Usage:
        hist_loss_fn = RGBuvHistogramLoss()
        loss = hist_loss_fn(stylized_image, style_image)
    
    CRITICAL: Official baseline.py uses:
        self.hist = RGBuvHistBlock(insz=64, h=256, intensity_scale=True, method='inverse-quadratic')
    """
    
    def __init__(self, h=256, insz=64, method='inverse-quadratic', sigma=0.02):
        """
        Initialize histogram loss module.
        
        Args:
            h: Number of histogram bins (default: 256 - MATCHES OFFICIAL)
            insz: Input size for resizing (default: 64 - MATCHES OFFICIAL)
            method: Histogram kernel method
            sigma: Bandwidth for soft binning
        """
        super(RGBuvHistogramLoss, self).__init__()
        self.hist_block = RGBuvHistBlock(insz=insz, h=h, intensity_scale=True, method=method, sigma=sigma)
    
    def forward(self, input, target):
        """
        Compute histogram loss between input and target images.
        
        Args:
            input: Input image (B, 3, H, W) in [0, 1]
            target: Target image (B, 3, H, W) in [0, 1]
            
        Returns:
            Histogram loss scalar
        """
        input_hist = self.hist_block(input)
        target_hist = self.hist_block(target)
        loss = histogram_loss(input_hist, target_hist)
        return loss
