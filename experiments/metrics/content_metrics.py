"""
xAILab Bamberg
University of Bamberg

@description:
Content-based metrics for evaluating color transfer methods.
Enhanced version with multiple edge detection and depth estimation models,
using Hugging Face transformers pipeline for depth estimation.
"""

import os
import numpy as np
import torch
import torchvision
from skimage.metrics import structural_similarity as ssim
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image

# Global model instances (lazy-loaded)
_lpips_model = None
_depth_models = {}  # Dictionary to store depth estimation pipelines
_hed_model = None
_ldc_model = None
_dists_model = None
_adists_model = None


EDGE_DETECTION_MODELS_DIR = Path(__file__).parent / "edge_models"

# Depth model configurations for transformers pipeline
DEPTH_MODEL_CONFIGS = {
    'depthpro': {
        'model_name': 'apple/DepthPro-hf',
        'processor_class': 'DepthProImageProcessorFast',
        'model_class': 'DepthProForDepthEstimation',
    },
    'depthanything_v2_large': {
        'model_name': 'depth-anything/Depth-Anything-V2-Large-hf',
        'processor_class': 'AutoImageProcessor',
        'model_class': 'AutoModelForDepthEstimation',
    },
    'dpt_large': {
        'model_name': 'Intel/dpt-large',
        'processor_class': 'AutoImageProcessor',
        'model_class': 'AutoModelForDepthEstimation',
    },
}


def set_path_to_edge_model(edge_dir: Optional[str] = None) -> None:
    """
    Set custom model directory paths.
    
    Args:
        edge_dir: Path to edge detection models directory
    """
    global EDGE_DETECTION_MODELS_DIR
    
    if edge_dir is not None:
        EDGE_DETECTION_MODELS_DIR = Path(edge_dir)
        EDGE_DETECTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def list_available_depth_models() -> list:
    """
    List all available depth estimation models.
    
    Returns:
        List of model names that can be used with depth estimation functions
    """
    return list(DEPTH_MODEL_CONFIGS.keys())


def _get_lpips_model():
    """
    Get the LPIPS model for perceptual distance calculation.
    Lazily loads the model on first use.
    """
    global _lpips_model
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net='alex')
            if torch.cuda.is_available():
                _lpips_model = _lpips_model.cuda()
            _lpips_model.eval()
        except ImportError:
            raise ImportError("Please install lpips package: pip install lpips")
    return _lpips_model


def _get_hed_model():
    """
    Get the HED (Holistically-Nested Edge Detection) model.
    Uses implementation from https://github.com/sniklaus/pytorch-hed
    Lazily loads the model on first use.
    """
    global _hed_model
    if _hed_model is None:
        try:
            # Import HED implementation
            from experiments.metrics.edge_models.hed import Network
            
            _hed_model = Network()
            
            # Load pretrained weights if available
            model_path = EDGE_DETECTION_MODELS_DIR / "hed.pth"
            if model_path.exists():
                # Load checkpoint (weights_only=False for compatibility with official weights)
                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
                
                # Extract state dict if checkpoint contains metadata
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Rename keys: module -> net (for compatibility with official weights)
                state_dict = {key.replace('module', 'net'): value for key, value in state_dict.items()}
                
                _hed_model.load_state_dict(state_dict)
                print(f"HED model loaded from {model_path.name}")
            else:
                print(f"HED weights not found at {model_path.absolute()}, using untrained model")
            
            if torch.cuda.is_available():
                _hed_model = _hed_model.cuda()
            _hed_model.eval()
        except Exception as e:
            raise RuntimeError(f"Could not load HED model: {e}")
    return _hed_model


def _get_ldc_model():
    """
    Get the LDC (Lightweight Dense CNN for Edge Detection) model.
    Uses implementation from NeuralPreset repository.
    Lazily loads the model on first use.
    """
    global _ldc_model
    if _ldc_model is None:
        try:
            # Import LDC implementation
            from experiments.metrics.edge_models.ldc import LDC
            
            _ldc_model = LDC()
            model_path = EDGE_DETECTION_MODELS_DIR / "ldc_biped.pth"
            
            if model_path.exists():
                # Load checkpoint (weights_only=False for compatibility)
                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
                
                # Extract state dict if checkpoint contains metadata
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                _ldc_model.load_state_dict(state_dict)
                print(f"LDC model loaded from {model_path.name}")
            else:
                print(f"LDC weights not found at {model_path.absolute()}, using untrained model")
            
            if torch.cuda.is_available():
                _ldc_model = _ldc_model.cuda()
            _ldc_model.eval()
        except Exception as e:
            raise RuntimeError(f"Could not initialize LDC model: {e}")
    return _ldc_model


def _get_depth_model(model_name: str) -> Dict[str, Any]:
    """
    Get or create a depth estimation model using transformers pipeline.
    
    Args:
        model_name: Name of the depth model ('depthpro', 'depthanything_v2_large', 'dpt_large')
        
    Returns:
        Dictionary containing the model pipeline and processor
        
    Reference:
        - DepthPro: https://huggingface.co/apple/DepthPro-hf
        - Depth Anything V2: https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf
        - DPT-Large: https://huggingface.co/Intel/dpt-large
    """
    global _depth_models
    
    if model_name not in _depth_models:
        if model_name not in DEPTH_MODEL_CONFIGS:
            raise ValueError(f"Unknown depth model: {model_name}. Choose from {list(DEPTH_MODEL_CONFIGS.keys())}")
        
        config = DEPTH_MODEL_CONFIGS[model_name]
        
        try:
            from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
            from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
            
            device = 0 if torch.cuda.is_available() else -1
            
            # Create pipeline
            print(f"Loading {model_name} depth estimation model from {config['model_name']}...")
            depth_pipe = pipeline(
                task="depth-estimation",
                model=config['model_name'],
                device=device
            )
            
            # Also load processor and model separately for more control
            if config['processor_class'] == 'DepthProImageProcessorFast':
                processor = DepthProImageProcessorFast.from_pretrained(config['model_name'])
                model = DepthProForDepthEstimation.from_pretrained(config['model_name'])
            else:
                processor = AutoImageProcessor.from_pretrained(config['model_name'])
                model = AutoModelForDepthEstimation.from_pretrained(config['model_name'])
            
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            
            _depth_models[model_name] = {
                'pipeline': depth_pipe,
                'processor': processor,
                'model': model,
                'config': config
            }
            
            print(f"{model_name} depth model loaded successfully")
            
        except ImportError as e:
            print(f"Failed to import required transformers modules: {e}")
            print("Please install: pip install transformers")
            _depth_models[model_name] = None
        except Exception as e:
            print(f"Could not load {model_name} depth model: {e}")
            _depth_models[model_name] = None
    
    return _depth_models[model_name]


def _sobel_edge_detection(img: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection to an image.
    
    Args:
        img: Image as numpy array in range [0,1]
        
    Returns:
        Edge map as numpy array
    """
    from scipy import ndimage
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    else:
        gray = img
    
    # Compute gradients
    sobelx = ndimage.sobel(gray, axis=0)
    sobely = ndimage.sobel(gray, axis=1)
    
    # Compute magnitude
    edges = np.hypot(sobelx, sobely)
    
    # Normalize
    if edges.max() > 0:
        edges = edges / edges.max()
    
    return edges


def _get_dists_model():
    """
    Get the DISTS (Deep Image Structure and Texture Similarity) metric model.
    Lazily loads the model on first use.
    """
    global _dists_model
    if _dists_model is None:
        try:
            from DISTS_pytorch import DISTS
            _dists_model = DISTS()
            if torch.cuda.is_available():
                _dists_model = _dists_model.cuda()
            print("DISTS model loaded successfully")
        except ImportError:
            print("DISTS not installed. Install with: pip install dists-pytorch")
            _dists_model = None
        except Exception as e:
            print(f"Could not load DISTS model: {e}")
            _dists_model = None
    return _dists_model


def _get_adists_model():
    """
    Get the A-DISTS (Adaptive DISTS) metric model.
    Lazily loads the model on first use.
    """
    global _adists_model
    if _adists_model is None:
        try:
            # Import A-DISTS implementation (need to add this to the codebase)
            from experiments.metrics.similarity_models.adists import ADISTS
            _adists_model = ADISTS()
            if torch.cuda.is_available():
                _adists_model = _adists_model.cuda()
            print("A-DISTS model loaded successfully")
        except Exception as e:
            print(f"Could not load A-DISTS model: {e}")
            _adists_model = None
    return _adists_model


def compute_luminance_ssim(img1, img2) -> float:
    """
    Compute SSIM on the luminance channel.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        
    Returns:
        Luminance SSIM
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
    
    # Convert to grayscale using standard luminance formula (on GPU if available)
    gray1 = 0.2989 * img1[0] + 0.5870 * img1[1] + 0.1140 * img1[2]
    gray2 = 0.2989 * img2[0] + 0.5870 * img2[1] + 0.1140 * img2[2]
    
    # Convert to numpy for skimage SSIM (requires numpy)
    gray1_np = gray1.cpu().numpy()
    gray2_np = gray2.cpu().numpy()
    
    # Compute SSIM
    return float(ssim(gray1_np, gray2_np, data_range=1.0))


def compute_ssim(img1, img2) -> float:
    """
    Compute SSIM (Structural Similarity Index Measure) on RGB channels.
    
    This metric assesses how well the structure of the original image is preserved.
    Higher values indicate better structural preservation.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        
    Returns:
        RGB SSIM score (range [0, 1], higher is better)
        
    Reference:
        Wang et al. "Image Quality Assessment: From Error Visibility to 
        Structural Similarity." IEEE TIP 2004.
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
    
    # Convert to numpy for skimage SSIM (requires HWC format)
    img1_np = img1.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img2_np = img2.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    
    # Compute SSIM on RGB channels
    ssim_value = ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)
    
    return float(ssim_value)


def compute_lpips_distance(img1, img2) -> float:
    """
    Compute LPIPS perceptual distance.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        
    Returns:
        LPIPS distance
    """
    # Get the LPIPS model
    model = _get_lpips_model()
    
    # Ensure inputs are tensors [C, H, W]
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1).float()
        if img1.ndim == 3 and img1.shape[-1] == 3:
            img1 = img1.permute(2, 0, 1)
    
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2).float()
        if img2.ndim == 3 and img2.shape[-1] == 3:
            img2 = img2.permute(2, 0, 1)
    
    # LPIPS expects tensors in range [-1, 1] with shape [B,C,H,W]
    img1_t = img1.unsqueeze(0) * 2 - 1
    img2_t = img2.unsqueeze(0) * 2 - 1
    
    if torch.cuda.is_available():
        img1_t = img1_t.cuda()
        img2_t = img2_t.cuda()
    
    # Compute distance
    with torch.no_grad():
        lpips_dist = model(img1_t, img2_t)
    
    return float(lpips_dist.item())


def _get_edges(img, method: str = 'ldc') -> torch.Tensor:
    """
    Extract edge map from an image using the specified method.
    
    Args:
        img: Image as torch.Tensor [C, H, W] in range [0,1]
        method: Edge detection method ('hed', 'ldc', 'teed', or 'sobel')
        
    Returns:
        Edge map as torch.Tensor [1, H, W] in range [0,1]
    """
    # Ensure input is tensor [C, H, W]
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img).float()
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
            
    # Ensure float32 to avoid type mismatch errors with models
    img = img.float()
    
    if method.lower() == 'sobel':
        # Sobel requires numpy - convert temporarily
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        edges_np = _sobel_edge_detection(img_np)
        return torch.from_numpy(edges_np).float().unsqueeze(0)  # [1, H, W]
    
    elif method.lower() == 'hed':
        model = _get_hed_model()
        if model is None:
            print("HED model not available, falling back to Sobel")
            # Fallback to Sobel
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            edges_np = _sobel_edge_detection(img_np)
            return torch.from_numpy(edges_np).float().unsqueeze(0)  # [1, H, W]
        
        try:
            # Prepare input tensor (HED expects input in range [0, 1])
            img_tensor = img.unsqueeze(0)  # [1, C, H, W]
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            with torch.no_grad():
                # HED forward returns [1, 1, H, W] with edges already in [0, 1]
                edges_tensor = model(img_tensor)
                edges = edges_tensor.squeeze(0)  # [1, H, W], keep channel dimension
            
            return edges.cpu()
        except Exception as e:
            print(f"HED processing failed: {e}, falling back to Sobel")
            # Fallback to Sobel
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            edges_np = _sobel_edge_detection(img_np)
            return torch.from_numpy(edges_np).float().unsqueeze(0)  # [1, H, W]
    
    elif method.lower() == 'ldc':
        model = _get_ldc_model()
        if model is None:
            print(f"{method.upper()} model not available, falling back to Sobel")
            # Fallback to Sobel
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            edges_np = _sobel_edge_detection(img_np)
            return torch.from_numpy(edges_np).float().unsqueeze(0)  # [1, H, W]
        
        try:
            # Prepare input tensor
            img_tensor = img.unsqueeze(0) * 255  # [1, C, H, W]
            
            mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
            if img_tensor.is_cuda:
                mean = mean.cuda()
            img_tensor = img_tensor - mean
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            with torch.no_grad():
                edges_list = model(img_tensor)
                # Both LDC and TEED return a list, use the fused output (last one)
                edges = edges_list[-1]
                edges = torch.sigmoid(edges).squeeze(0)  # [1, H, W], keep channel dimension
            
            # Normalize
            edges_max = edges.max()
            if edges_max > 0:
                edges = edges / edges_max
            
            return edges.cpu()
        except Exception as e:
            print(f"{method.upper()} processing failed: {e}, falling back to Sobel")
            # Fallback to Sobel
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            edges_np = _sobel_edge_detection(img_np)
            return torch.from_numpy(edges_np).float().unsqueeze(0)  # [1, H, W]
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}. Choose from 'hed', 'ldc', 'teed', or 'sobel'.")


def _get_depth_map(img, method: str = 'depthanything_v2_large') -> torch.Tensor:
    """
    Extract depth map from an image using the specified method via Hugging Face transformers.
    
    Args:
        img: Image as torch.Tensor [C, H, W] in range [0,1]
        method: Depth estimation method. Options:
                - 'depthpro': Apple DepthPro (sharp metric depth)
                - 'depthanything_v2_large': Depth Anything V2 Large (state-of-the-art relative depth)
                - 'dpt_large': DPT-Large / MiDaS 3.0
        
    Returns:
        Depth map as torch.Tensor [1, H, W] with normalized values
        
    References:
        - DepthPro: https://huggingface.co/apple/DepthPro-hf
        - Depth Anything V2: https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf
        - DPT-Large: https://huggingface.co/Intel/dpt-large
    """
    model_dict = _get_depth_model(method)
    if model_dict is None:
        raise RuntimeError(f"{method} depth model not available")
    
    try:
        # Ensure input is tensor [C, H, W]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
        
        # Convert to numpy for PIL (transformers requires PIL input)
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8)
        
        # Get original image size
        original_size = (pil_image.height, pil_image.width)
        
        # Use the pipeline for inference
        pipeline_obj = model_dict['pipeline']
        processor = model_dict['processor']
        model = model_dict['model']
        
        # For more control, use processor and model directly
        inputs = processor(images=pil_image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process the output
        if method == 'depthpro':
            # DepthPro has special post-processing
            post_processed = processor.post_process_depth_estimation(
                outputs,
                target_sizes=[original_size],
            )
            predicted_depth = post_processed[0]["predicted_depth"]
        else:
            # Standard post-processing for other models
            predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=original_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)  # [1, H, W], keep channel dimension
        
        # Keep as tensor
        depth_map = predicted_depth.cpu()
        
        # Normalize to [0, 1] range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = torch.zeros_like(depth_map)
        
        return depth_map
        
    except Exception as e:
        raise RuntimeError(f"{method} depth estimation failed: {e}")


def compute_edge_similarity(img1, img2, method: str = 'ldc') -> float:
    """
    Compute similarity between edge maps using SSIM.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Edge detection method ('hed', 'ldc', 'teed', or 'sobel'). Default is 'ldc'.
        
    Returns:
        Edge map SSIM similarity
    """
    try:
        # Get edge maps (returns torch tensors [1, H, W])
        edges1 = _get_edges(img1, method=method)
        edges2 = _get_edges(img2, method=method)
        
        # Squeeze and convert to numpy for skimage SSIM (requires 2D numpy arrays)
        edges1_np = edges1.squeeze(0).cpu().numpy()
        edges2_np = edges2.squeeze(0).cpu().numpy()
        
        # Compute SSIM between edge maps
        edge_ssim = ssim(edges1_np, edges2_np, data_range=1.0)
        return float(edge_ssim)
    except Exception as e:
        # Fallback to Sobel if the specified method fails
        print(f"Edge similarity ({method}) failed: {e}. Using Sobel fallback.")
        edges1 = _get_edges(img1, method='sobel')
        edges2 = _get_edges(img2, method='sobel')
        edges1_np = edges1.squeeze(0).cpu().numpy()
        edges2_np = edges2.squeeze(0).cpu().numpy()
        edge_ssim = ssim(edges1_np, edges2_np, data_range=1.0)
        return float(edge_ssim)


def compute_depth_consistency(img1, img2, method: str = 'depthanything_v2_large') -> float:
    """
    Compute consistency between depth maps using Spearman correlation.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Depth estimation method. Options:
                - 'depthpro': Apple DepthPro
                - 'depthanything_v2_large': Depth Anything V2 Large (default)
                - 'dpt_large': DPT-Large / MiDaS 3.0
        
    Returns:
        Spearman correlation between depth maps (higher is better)
    """
    try:
        # Get depth maps (returns torch tensors [1, H, W])
        depth1 = _get_depth_map(img1, method=method)
        depth2 = _get_depth_map(img2, method=method)
        
        # Normalize depths
        depth1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min() + 1e-8)
        depth2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min() + 1e-8)
        
        # Squeeze and convert to numpy for scipy spearmanr (requires 1D numpy arrays)
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(depth1_norm.squeeze(0).flatten().cpu().numpy(), depth2_norm.squeeze(0).flatten().cpu().numpy())
        
        # Convert NaN to 0
        if np.isnan(correlation):
            correlation = 0.0
            
        return float(correlation)
    except Exception as e:
        print(f"Depth consistency ({method}) failed: {e}")
        return 0.5


def compute_edge_dists(img1, img2, method: str = 'ldc') -> float:
    """
    Compute DISTS metric between edge maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Edge detection method ('hed', 'ldc', 'teed', or 'sobel'). Default is 'ldc'.
        
    Returns:
        DISTS distance between edge maps (lower is better)
    """
    model = _get_dists_model()
    if model is None:
        print("DISTS model not available, returning default value")
        return 0.5
    
    try:
        # Get edge maps (returns torch tensors [1, H, W])
        edges1 = _get_edges(img1, method=method)
        edges2 = _get_edges(img2, method=method)
        
        # Convert to RGB tensors (DISTS expects 3-channel input) [3, H, W]
        edges1_rgb = edges1.repeat(3, 1, 1)
        edges2_rgb = edges2.repeat(3, 1, 1)
        
        # Add batch dimension [1, 3, H, W]
        edges1_t = edges1_rgb.unsqueeze(0)
        edges2_t = edges2_rgb.unsqueeze(0)
        
        if torch.cuda.is_available():
            edges1_t = edges1_t.cuda()
            edges2_t = edges2_t.cuda()
        
        # Compute DISTS
        with torch.no_grad():
            dists_value = model(edges1_t, edges2_t)
        
        return float(dists_value.item())
    except Exception as e:
        print(f"Edge DISTS calculation failed: {e}")
        return 0.5


def compute_edge_adists(img1, img2, method: str = 'ldc') -> float:
    """
    Compute A-DISTS metric between edge maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Edge detection method ('hed', 'ldc', 'teed', or 'sobel'). Default is 'ldc'.
        
    Returns:
        A-DISTS distance between edge maps (lower is better)
    """
    model = _get_adists_model()
    if model is None:
        print("A-DISTS model not available, returning default value")
        return 0.5
    
    try:
        # Get edge maps (returns torch tensors [1, H, W])
        edges1 = _get_edges(img1, method=method)
        edges2 = _get_edges(img2, method=method)
        
        # Convert to RGB tensors (A-DISTS expects 3-channel input) [3, H, W]
        edges1_rgb = edges1.repeat(3, 1, 1)
        edges2_rgb = edges2.repeat(3, 1, 1)
        
        # Add batch dimension [1, 3, H, W]
        edges1_t = edges1_rgb.unsqueeze(0)
        edges2_t = edges2_rgb.unsqueeze(0)
        
        if torch.cuda.is_available():
            edges1_t = edges1_t.cuda()
            edges2_t = edges2_t.cuda()
        
        # Compute A-DISTS
        with torch.no_grad():
            adists_value = model(edges1_t, edges2_t)
        
        return float(adists_value.item())
    except Exception as e:
        print(f"Edge A-DISTS calculation failed: {e}")
        return 0.5


def compute_depth_dists(img1, img2, method: str = 'depthanything_v2_large') -> float:
    """
    Compute DISTS metric between depth maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Depth estimation method. Options:
                - 'depthpro': Apple DepthPro
                - 'depthanything_v2_large': Depth Anything V2 Large (default)
                - 'dpt_large': DPT-Large / MiDaS 3.0
        
    Returns:
        DISTS distance between depth maps (lower is better)
    """
    model = _get_dists_model()
    if model is None:
        print("DISTS model not available, returning default value")
        return 0.5
    
    try:
        # Get depth maps (returns torch tensors [1, H, W])
        depth1 = _get_depth_map(img1, method=method)
        depth2 = _get_depth_map(img2, method=method)
        
        # Normalize depths
        depth1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min() + 1e-8)
        depth2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min() + 1e-8)
        
        # Convert to RGB tensors (DISTS expects 3-channel input) [3, H, W]
        depth1_rgb = depth1_norm.repeat(3, 1, 1)
        depth2_rgb = depth2_norm.repeat(3, 1, 1)
        
        # Add batch dimension [1, 3, H, W]
        depth1_t = depth1_rgb.unsqueeze(0)
        depth2_t = depth2_rgb.unsqueeze(0)
        
        if torch.cuda.is_available():
            depth1_t = depth1_t.cuda()
            depth2_t = depth2_t.cuda()
        
        # Compute DISTS
        with torch.no_grad():
            dists_value = model(depth1_t, depth2_t)
        
        return float(dists_value.item())
    except Exception as e:
        print(f"Depth DISTS calculation failed: {e}")
        return 0.5


def compute_depth_adists(img1, img2, method: str = 'depthanything_v2_large') -> float:
    """
    Compute A-DISTS metric between depth maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Depth estimation method. Options:
                - 'depthpro': Apple DepthPro
                - 'depthanything_v2_large': Depth Anything V2 Large (default)
                - 'dpt_large': DPT-Large / MiDaS 3.0
        
    Returns:
        A-DISTS distance between depth maps (lower is better)
    """
    model = _get_adists_model()
    if model is None:
        print("A-DISTS model not available, returning default value")
        return 0.5
    
    try:
        # Get depth maps (returns torch tensors [1, H, W])
        depth1 = _get_depth_map(img1, method=method)
        depth2 = _get_depth_map(img2, method=method)
        
        # Normalize depths
        depth1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min() + 1e-8)
        depth2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min() + 1e-8)
        
        # Convert to RGB tensors (A-DISTS expects 3-channel input) [3, H, W]
        depth1_rgb = depth1_norm.repeat(3, 1, 1)
        depth2_rgb = depth2_norm.repeat(3, 1, 1)
        
        # Add batch dimension [1, 3, H, W]
        depth1_t = depth1_rgb.unsqueeze(0)
        depth2_t = depth2_rgb.unsqueeze(0)
        
        if torch.cuda.is_available():
            depth1_t = depth1_t.cuda()
            depth2_t = depth2_t.cuda()
        
        # Compute A-DISTS
        with torch.no_grad():
            adists_value = model(depth1_t, depth2_t)
        
        return float(adists_value.item())
    except Exception as e:
        print(f"Depth A-DISTS calculation failed: {e}")
        return 0.5


def compute_edge_fom(img1, img2, method: str = 'ldc') -> float:
    """
    Compute Pratt's Figure of Merit (FOM) for edge maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Edge detection method ('hed', 'ldc', 'teed', or 'sobel'). Default is 'ldc'.
        
    Returns:
        Pratt's FOM score (range [0, 1], higher is better)
    """
    from scipy import ndimage
    
    try:
        # Get edge maps (returns torch tensors [1, H, W])
        edges1 = _get_edges(img1, method=method)
        edges2 = _get_edges(img2, method=method)
        
        # Convert to numpy and binarize
        # Using 0.1 threshold as edge maps are normalized [0,1]
        edges1_np = (edges1.squeeze(0).cpu().numpy() > 0.1)
        edges2_np = (edges2.squeeze(0).cpu().numpy() > 0.1)
        
        # Helper counts
        N_I = np.sum(edges1_np)
        N_A = np.sum(edges2_np)
        
        if N_I == 0:
            return 0.0
        
        # Compute distance transform on reference (img1)
        # We need distance from detected edge points to nearest reference edge point
        # Distance transform is computed on background, so invert reference
        distances = ndimage.distance_transform_edt(~edges1_np)
        
        # Get distances for detected edge points
        d_i = distances[edges2_np]
        
        # Scaling constant alpha (typically 1/9)
        alpha = 1.0 / 9.0
        
        # Compute FOM
        # Sum(1 / (1 + alpha * d^2)) / max(N_I, N_A)
        fom_sum = np.sum(1.0 / (1.0 + alpha * (d_i ** 2)))
        fom = fom_sum / max(N_I, N_A) if max(N_I, N_A) > 0 else 0.0
        
        return float(fom)
    except Exception as e:
        print(f"Edge FOM calculation failed: {e}")
        return 0.0


def compute_edge_hausdorff(img1, img2, method: str = 'ldc') -> float:
    """
    Compute Hausdorff Distance for edge maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Edge detection method ('hed', 'ldc', 'teed', or 'sobel'). Default is 'ldc'.
        
    Returns:
        Hausdorff distance (lower is better)
    """
    from scipy.spatial.distance import directed_hausdorff
    
    try:
        # Get edge maps (returns torch tensors [1, H, W])
        edges1 = _get_edges(img1, method=method)
        edges2 = _get_edges(img2, method=method)
        
        # Convert to numpy and binarize
        edges1_np = (edges1.squeeze(0).cpu().numpy() > 0.1)
        edges2_np = (edges2.squeeze(0).cpu().numpy() > 0.1)
        
        if not np.any(edges1_np) or not np.any(edges2_np):
            # Return a large value if no edges are found
            return 1000.0
            
        # Get coordinates of edge pixels
        pts1 = np.argwhere(edges1_np)
        pts2 = np.argwhere(edges2_np)
        
        # Compute directed Hausdorff distances
        d1 = directed_hausdorff(pts1, pts2)[0]
        d2 = directed_hausdorff(pts2, pts1)[0]
        
        return float(max(d1, d2))
    except Exception as e:
        print(f"Edge Hausdorff calculation failed: {e}")
        return 1000.0


def compute_depth_mae(img1, img2, method: str = 'depthanything_v2_large') -> float:
    """
    Compute Mean Absolute Error (MAE) between depth maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Depth estimation method. Options same as compute_depth_consistency.
        
    Returns:
        MAE between normalized depth maps (lower is better)
    """
    try:
        # Get depth maps (returns torch tensors [1, H, W])
        depth1 = _get_depth_map(img1, method=method)
        depth2 = _get_depth_map(img2, method=method)
        
        # Compute MAE
        mae = torch.mean(torch.abs(depth1 - depth2))
        
        return float(mae.item())
    except Exception as e:
        print(f"Depth MAE calculation failed: {e}")
        return 1.0


def compute_depth_ssim(img1, img2, method: str = 'depthanything_v2_large') -> float:
    """
    Compute SSIM between depth maps.
    
    Args:
        img1: First image as torch.Tensor [C, H, W] in range [0,1]
        img2: Second image as torch.Tensor [C, H, W] in range [0,1]
        method: Depth estimation method. Options same as compute_depth_consistency.
        
    Returns:
        SSIM between depth maps (higher is better)
    """
    try:
        # Get depth maps (returns torch tensors [1, H, W])
        depth1 = _get_depth_map(img1, method=method)
        depth2 = _get_depth_map(img2, method=method)
        
        # Convert to numpy for ssim
        depth1_np = depth1.squeeze(0).cpu().numpy()
        depth2_np = depth2.squeeze(0).cpu().numpy()
        
        ssim_val = ssim(depth1_np, depth2_np, data_range=1.0)
        
        return float(ssim_val)
    except Exception as e:
        print(f"Depth SSIM calculation failed: {e}")
        return 0.0


