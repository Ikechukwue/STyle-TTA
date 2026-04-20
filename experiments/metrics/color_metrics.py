"""
xAILab Bamberg
University of Bamberg

@description:
Color-based metrics for evaluating color transfer methods.
"""

import numpy as np
import scipy.stats
from skimage import color
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
from pathlib import Path
import warnings

# Global model storage
_fid_model = None
_vgg_model = None

def _convert_to_lab(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to LAB color space.
    
    Args:
        img: RGB image as numpy array in range [0,1]
        
    Returns:
        LAB image as numpy array
    """
    return color.rgb2lab(img)


def compute_wasserstein_distance(img1, img2, 
                                color_space: str = 'lab') -> float:
    """
    Compute Wasserstein (Earth Mover's) distance between color distributions.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        color_space: Color space to use ('rgb', 'lab', etc.')
        
    Returns:
        Wasserstein distance
    """
    # Ensure inputs are tensors [C, H, W]
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
        if img1.ndim == 3 and img1.shape[-1] == 3:
            img1 = img1.permute(2, 0, 1)
    
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
        if img2.ndim == 3 and img2.shape[-1] == 3:
            img2 = img2.permute(2, 0, 1)
    
    # Convert to numpy for color space conversion and scipy (requires numpy)
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)
    
    # Convert to appropriate color space if needed
    if color_space.lower() == 'lab':
        img1_cs = _convert_to_lab(img1_np)
        img2_cs = _convert_to_lab(img2_np)
    else:
        # Use as-is for RGB or other spaces that don't need conversion
        img1_cs = img1_np
        img2_cs = img2_np
    
    # Reshape to 2D arrays (pixels x channels)
    pixels1 = img1_cs.reshape(-1, img1_cs.shape[-1])
    pixels2 = img2_cs.reshape(-1, img2_cs.shape[-1])
    
    # Compute Wasserstein distance for each channel
    distances = []
    for channel in range(pixels1.shape[1]):
        # Get distributions for this channel
        dist1 = pixels1[:, channel]
        dist2 = pixels2[:, channel]
        
        # Compute 1D Wasserstein distance (Earth Mover's Distance)
        # Using SciPy's implementation
        distances.append(scipy.stats.wasserstein_distance(dist1, dist2))
    
    # Average across channels
    return float(np.mean(distances))


def _compute_histogram(img, bins: int = 256) -> List[np.ndarray]:
    """
    Compute histograms for each channel.
    
    Args:
        img: Image as torch.Tensor [C, H, W] in range [0,1]
        bins: Number of histogram bins
        
    Returns:
        List of histograms for each channel
    """
    # Ensure input is tensor [C, H, W]
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img).float()
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
    
    histograms = []
    for channel in range(img.shape[0]):
        # Convert to numpy for histogram computation (numpy is more efficient for this)
        channel_data = img[channel].cpu().numpy().flatten()
        hist, _ = np.histogram(channel_data, bins=bins, range=(0, 1), density=True)
        histograms.append(hist)
    return histograms


def compute_histogram_distance(img1, img2, 
                              bins: int = 256) -> Tuple[float, float, float, float]:
    """
    Compute various histogram distance metrics between two images.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        bins: Number of histogram bins
        
    Returns:
        Tuple of (KL divergence, JS divergence, Chi-square distance, Histogram intersection)
    """
    # Compute histograms
    hist1 = _compute_histogram(img1, bins)
    hist2 = _compute_histogram(img2, bins)
    
    kl_divs = []
    js_divs = []
    chi_squares = []
    intersections = []
    
    for h1, h2 in zip(hist1, hist2):
        # Add small epsilon to avoid division by zero
        h1_safe = h1 + 1e-10
        h2_safe = h2 + 1e-10
        
        # Normalize
        h1_safe = h1_safe / np.sum(h1_safe)
        h2_safe = h2_safe / np.sum(h2_safe)
        
        # KL divergence: sum(p(i) * log(p(i)/q(i)))
        kl_div = np.sum(h1_safe * np.log(h1_safe / h2_safe))
        kl_divs.append(kl_div)
        
        # JS divergence: 0.5 * (KL(p|m) + KL(q|m)) where m = 0.5 * (p + q)
        m = 0.5 * (h1_safe + h2_safe)
        js_div = 0.5 * (np.sum(h1_safe * np.log(h1_safe / m)) + np.sum(h2_safe * np.log(h2_safe / m)))
        js_divs.append(js_div)
        
        # Chi-square: sum((p(i) - q(i))^2 / q(i))
        chi_square = np.sum((h1_safe - h2_safe)**2 / h2_safe)
        chi_squares.append(chi_square)
        
        # Histogram intersection: sum(min(p(i), q(i)))
        intersection = np.sum(np.minimum(h1_safe, h2_safe))
        intersections.append(intersection)
    
    # Average across channels
    return (
        float(np.mean(kl_divs)),
        float(np.mean(js_divs)),
        float(np.mean(chi_squares)),
        float(np.mean(intersections))
    )


def compute_color_moment_distance(img1, img2) -> float:
    """
    Compute color moment distance (mean and covariance differences).
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        
    Returns:
        Color moment distance
    """
    # Ensure inputs are tensors [C, H, W]
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
        if img1.ndim == 3 and img1.shape[-1] == 3:
            img1 = img1.permute(2, 0, 1)
    
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
        if img2.ndim == 3 and img2.shape[-1] == 3:
            img2 = img2.permute(2, 0, 1)
    
    # Reshape to 2D arrays (channels x pixels)
    pixels1 = img1.reshape(img1.shape[0], -1).T  # [pixels, channels]
    pixels2 = img2.reshape(img2.shape[0], -1).T  # [pixels, channels]
    
    # Convert to numpy for covariance computation (numpy.cov is more stable)
    pixels1_np = pixels1.cpu().numpy()
    pixels2_np = pixels2.cpu().numpy()
    
    # Compute means
    mean1 = np.mean(pixels1_np, axis=0)
    mean2 = np.mean(pixels2_np, axis=0)
    
    # Compute covariances
    cov1 = np.cov(pixels1_np, rowvar=False)
    cov2 = np.cov(pixels2_np, rowvar=False)
    
    # Mean difference
    mean_diff = np.linalg.norm(mean1 - mean2)
    
    # Covariance difference (Frobenius norm)
    cov_diff = np.linalg.norm(cov1 - cov2, ord='fro')
    
    # Weighted sum
    return float(0.5 * mean_diff + 0.5 * cov_diff)


def _get_fid_model():
    """
    Get FID (Fréchet Inception Distance) model.
    Uses InceptionV3 for feature extraction.
    """
    global _fid_model
    if _fid_model is None:
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            _fid_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            _fid_model.fc = nn.Identity()  # Remove final layer
            if torch.cuda.is_available():
                _fid_model = _fid_model.cuda()
            _fid_model.eval()
            print("Inception V3 model loaded for FID computation")
        except Exception as e:
            print(f"Could not load Inception V3 model: {e}")
            _fid_model = None
    return _fid_model


def _get_vgg_model():
    """
    Get VGG19 model for Gatys style similarity.
    """
    global _vgg_model
    if _vgg_model is None:
        try:
            from torchvision.models import vgg19, VGG19_Weights
            _vgg_model = vgg19(weights=VGG19_Weights.DEFAULT).features
            if torch.cuda.is_available():
                _vgg_model = _vgg_model.cuda()
            _vgg_model.eval()
            print("VGG19 model loaded for Gatys style similarity")
        except Exception as e:
            print(f"Could not load VGG19 model: {e}")
            _vgg_model = None
    return _vgg_model


def _calculate_activation_statistics(images: torch.Tensor, model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and covariance statistics of Inception activations.
    
    Args:
        images: Batch of images as tensor [B, C, H, W] in range [0, 1]
        model: Inception model
        
    Returns:
        Tuple of (mu, sigma) where mu is mean and sigma is covariance
    """
    model.eval()
    
    # Prepare images for Inception (expects [-1, 1] range and specific size)
    from torchvision.transforms import functional as F
    
    processed_images = []
    for img in images:
        # Resize to 299x299 (Inception input size)
        img_resized = F.resize(img, [299, 299], antialias=True)
        # Normalize to [-1, 1]
        img_normalized = img_resized * 2 - 1
        processed_images.append(img_normalized)
    
    batch = torch.stack(processed_images)
    
    # Get activations
    with torch.no_grad():
        activations = model(batch)
        
    # Convert to numpy
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    
    # Calculate statistics
    mu = np.mean(activations, axis=0)
    
    # Check if we have enough samples for covariance calculation
    # Need at least 2 samples, ideally many more than feature dimension
    n_samples = activations.shape[0]
    n_features = activations.shape[1] if len(activations.shape) > 1 else 1
    
    if n_samples < 2:
        # Not enough samples - use identity matrix scaled by variance
        warnings.warn(f"Only {n_samples} sample(s) for FID - need at least 2. Using identity covariance.")
        sigma = np.eye(n_features) * 1e-6
    elif n_samples <= n_features:
        # Fewer samples than features - covariance will be singular
        # Use regularized covariance
        warnings.warn(f"Only {n_samples} samples for {n_features} features in FID computation. "
                     f"Covariance matrix will be ill-conditioned. Consider using more samples.")
        sigma = np.cov(activations, rowvar=False)
        # Add small diagonal regularization to ensure positive definite
        sigma = sigma + np.eye(n_features) * 1e-6
    else:
        # Standard case
        sigma = np.cov(activations, rowvar=False)
    
    return mu, sigma


def _calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                                mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Calculate Fréchet distance between two Gaussian distributions.
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        
    Returns:
        Fréchet distance
    """
    import scipy.linalg
    
    # Calculate squared difference of means
    diff = mu1 - mu2
    
    # Product might be negative due to numerical errors
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_fid(images1: torch.Tensor, images2: torch.Tensor) -> float:
    """
    Compute Fréchet Inception Distance (FID) between two sets of images.
    
    FID measures the distance between two distributions of images by comparing
    their activation statistics in Inception V3 feature space.
    
    Args:
        images1: First set of images [B, C, H, W] in range [0, 1]
        images2: Second set of images [B, C, H, W] in range [0, 1]
        
    Returns:
        FID score (lower is better)
        
    Reference:
        Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to 
        a Local Nash Equilibrium." NeurIPS 2017.
    """
    model = _get_fid_model()
    if model is None:
        warnings.warn("FID model not available, returning default value")
        return 100.0
    
    try:
        # Ensure tensors are on the right device
        device = next(model.parameters()).device
        images1 = images1.to(device)
        images2 = images2.to(device)
        
        # Calculate statistics for both distributions
        mu1, sigma1 = _calculate_activation_statistics(images1, model)
        mu2, sigma2 = _calculate_activation_statistics(images2, model)
        
        # Calculate FID
        fid_value = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return float(fid_value)
    except Exception as e:
        warnings.warn(f"FID computation failed: {e}")
        return 100.0


def compute_artfid(fid_value: float, lpips_value: float) -> float:
    """
    Compute ArtFID (Artistic Fréchet Inception Distance) score.
    
    ArtFID combines style similarity (FID) with content preservation (LPIPS)
    to provide a comprehensive metric for neural style transfer evaluation.
    
    The formula is: ArtFID = (LPIPS + 1) * (FID + 1)
    
    Args:
        fid_value: FID score between style and stylized images
        lpips_value: LPIPS distance between content and stylized images
        
    Returns:
        ArtFID score (lower is better)
        
    Reference:
        Wright & Ommer. "ArtFID: Quantitative Evaluation of Neural Style Transfer." 
        GCPR 2022.
    """
    # Compute ArtFID using the simplified formula
    artfid = (lpips_value + 1.0) * (fid_value + 1.0)
    return float(artfid)


def _get_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for style representation.
    
    The Gram matrix captures style by computing correlations between feature maps.
    
    Args:
        features: Feature maps [B, C, H, W]
        
    Returns:
        Gram matrix [B, C, C]
    """
    B, C, H, W = features.shape
    features = features.view(B, C, H * W)
    
    # Compute Gram matrix: G = F * F^T / (C * H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram / (C * H * W)
    
    return gram


def _extract_vgg_features(images: torch.Tensor, model: nn.Module, 
                         layers: List[str] = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']) -> Dict[str, torch.Tensor]:
    """
    Extract features from specified VGG19 layers.
    
    Args:
        images: Input images [B, C, H, W] in range [0, 1]
        model: VGG19 features model
        layers: Layer names to extract features from
        
    Returns:
        Dictionary mapping layer names to feature tensors
    """
    # VGG normalization (ImageNet mean and std)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    if images.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    # Normalize
    normalized = (images - mean) / std
    
    # VGG layer mapping (0-indexed)
    layer_mapping = {
        'relu1_1': 1,
        'relu2_1': 6,
        'relu3_1': 11,
        'relu4_1': 20,
        'relu5_1': 29
    }
    
    features = {}
    x = normalized
    current_layer = 0
    
    # Process layers sequentially
    for name in layers:
        if name in layer_mapping:
            target_layer = layer_mapping[name]
            # Forward through layers from current position to target
            for i in range(current_layer, target_layer + 1):
                x = model[i](x)
            features[name] = x.clone()
            current_layer = target_layer + 1
    
    return features


def compute_gatys_style_loss(image1: torch.Tensor, image2: torch.Tensor,
                             layers: List[str] = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']) -> float:
    """
    Compute Gatys-style similarity using Gram matrix distance.
    
    This metric measures style similarity by comparing Gram matrices of VGG features,
    as proposed in the neural style transfer paper by Gatys et al.
    
    The style loss is computed as the mean squared error between Gram matrices across
    multiple VGG layers, averaged over all layers and batch samples.
    
    Args:
        image1: First image or batch [B, C, H, W] in range [0, 1]
        image2: Second image or batch [B, C, H, W] in range [0, 1]
        layers: VGG layers to use for style representation
        
    Returns:
        Style distance (lower is better, more similar style)
        
    Reference:
        Gatys, Ecker, Bethge. "Image Style Transfer Using Convolutional Neural Networks."
        CVPR 2016.
    """
    model = _get_vgg_model()
    if model is None:
        warnings.warn("VGG19 model not available, returning default value")
        return 1.0
    
    try:
        # Ensure tensors are on the right device
        device = next(model.parameters()).device
        image1 = image1.to(device)
        image2 = image2.to(device)
        
        # Extract features from both images
        with torch.no_grad():
            features1 = _extract_vgg_features(image1, model, layers)
            features2 = _extract_vgg_features(image2, model, layers)
        
        # Compute style loss as mean of Gram matrix MSE across layers
        style_losses = []
        for layer in layers:
            if layer in features1 and layer in features2:
                # Compute Gram matrices for this layer
                gram1 = _get_gram_matrix(features1[layer])
                gram2 = _get_gram_matrix(features2[layer])
                
                # MSE loss between Gram matrices (mean over all dimensions)
                # This computes mean((G1 - G2)^2) which is standard MSE
                layer_loss = torch.mean((gram1 - gram2) ** 2)
                style_losses.append(layer_loss)
        
        # Average style loss over all layers
        if len(style_losses) > 0:
            total_style_loss = torch.mean(torch.stack(style_losses))
            return float(total_style_loss.item())
        else:
            warnings.warn("No valid layers for Gatys style loss computation")
            return 1.0
        
    except Exception as e:
        warnings.warn(f"Gatys style loss computation failed: {e}")
        return 1.0