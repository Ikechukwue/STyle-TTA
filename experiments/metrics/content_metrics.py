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
import torch.nn.functional as F

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
    global _hed_model
    if _hed_model is None:
        from experiments.metrics.edge_models.hed import Network
        _hed_model = Network()
        model_path = EDGE_DETECTION_MODELS_DIR / "hed.pth"

        if model_path.exists():
            checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
            
            # Extract state dict if nested
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('net', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint

            # Define the manual mapping from Checkpoint -> Your Model Class
            mapping = {
                "conv1_1": "netVggOne.0", "conv1_2": "netVggOne.2",
                "conv2_1": "netVggTwo.1", "conv2_2": "netVggTwo.3",
                "conv3_1": "netVggThr.1", "conv3_2": "netVggThr.3", "conv3_3": "netVggThr.5",
                "conv4_1": "netVggFou.1", "conv4_2": "netVggFou.3", "conv4_3": "netVggFou.5",
                "conv5_1": "netVggFiv.1", "conv5_2": "netVggFiv.3", "conv5_3": "netVggFiv.5",
                "score_dsn1": "netScoreOne", "score_dsn2": "netScoreTwo",
                "score_dsn3": "netScoreThr", "score_dsn4": "netScoreFou",
                "score_dsn5": "netScoreFiv", "score_final": "netCombine.0"
            }

            new_state_dict = {}
            for old_key, value in state_dict.items():
                # Strip potential 'module.' or 'net.' prefixes first
                k = old_key.replace('module.', '').replace('net.', '')
                
                # Check if the base name (e.g., 'conv1_1') is in our map
                base_name = ".".join(k.split('.')[:-1])
                suffix = k.split('.')[-1] # weight or bias
                
                if base_name in mapping:
                    new_key = f"{mapping[base_name]}.{suffix}"
                    new_state_dict[new_key] = value
                else:
                    # If it's already named correctly or doesn't need mapping
                    new_state_dict[k] = value

            _hed_model.load_state_dict(new_state_dict)
            print(f"HED model successfully remapped and loaded.")
        
        if torch.cuda.is_available():
            _hed_model = _hed_model.cuda()
        _hed_model.eval()
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
            model_path = Path("/home/stud/nemmler/retristyle/data/models/edge_detection/ldc_biped.pth")
            
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

##############################################################################
# Model-Free Formulas
##############################################################################

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

##############################################################################
# Model Formulas
##############################################################################

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
    
    # Ensure inputs are tensors [B, C, H, W]
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

##############################################################################
# Batch Formulas
##############################################################################

def batch_ssim(X: torch.Tensor, Y: torch.Tensor, window_size: int = 11, size_average: bool = False, use_luminance: bool = False,) -> torch.Tensor:
    """
    Computes SSIM pairwise between batch X (N, C, H, W) and batch Y (M, C, H, W).
    Returns a matrix of shape (N, M) containing the SSIM for every pair.
    """
    
    # Create a 2D Gaussian window
    def gaussian(w_size, sigma, device, dtype):
        coords = torch.arange(w_size, device=device, dtype=dtype)
        coords -= w_size // 2

        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    def luminance(img1, img2):
        gray1 = (
            0.2989 * img1[:, 0:1]
            + 0.5870 * img1[:, 1:2]
            + 0.1140 * img1[:, 2:3]
        )
        gray2 = (
            0.2989 * img2[:, 0:1]
            + 0.5870 * img2[:, 1:2]
            + 0.1140 * img2[:, 2:3]
        )
        return gray1, gray2
    
    if use_luminance:
        X, Y = luminance(X,Y)

    N, C, H, W = X.shape
    M = Y.shape[0]
    
    _1D_window = gaussian(window_size, 1.5, X.device, X.dtype).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).to(dtype=X.dtype).unsqueeze(0).unsqueeze(0).to(X.device)
    window = _2D_window.expand(C, 1, window_size, window_size)

    # To do an N x M pairwise cross comparison efficiently, we expand the tensors
    # X_exp: (N, M, C, H, W) -> reshaped to (N*M, C, H, W)
    X_exp = X.unsqueeze(1).expand(N, M, C, H, W).reshape(N * M, C, H, W)
    Y_exp = Y.unsqueeze(0).expand(N, M, C, H, W).reshape(N * M, C, H, W)

    # Compute local means
    mu1 = F.conv2d(X_exp, window, groups=C, padding=window_size//2)
    mu2 = F.conv2d(Y_exp, window, groups=C, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariances
    sigma1_sq = F.conv2d(X_exp * X_exp, window, groups=C, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(Y_exp * Y_exp, window, groups=C, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(X_exp * Y_exp, window, groups=C, padding=window_size//2) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula applied across all NxM pairs at once
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Average over spatial dimensions and channels -> Shape (N * M) -> Reshape to (N, M)
    return ssim_map.mean(dim=[1, 2, 3]).reshape(N, M)


def batch_sobel_edge_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes Sobel Edge Magnitude similarity pairwise between batch X and batch Y.
    Returns a matrix of shape (N, M).
    """
    def rgb_to_gray(img):
        return (
            0.2989 * img[:, 0:1]
            + 0.5870 * img[:, 1:2]
            + 0.1140 * img[:, 2:3]
        )
    
    
    X = rgb_to_gray(X)
    Y = rgb_to_gray(Y)

    N, C, H, W = X.shape
    M = Y.shape[0]

    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=X.dtype).view(1, 1, 3, 3).to(X.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=X.dtype).view(1, 1, 3, 3).to(X.device)

    # Expand to match channels
    sobel_x = sobel_x.expand(C, 1, 3, 3)
    sobel_y = sobel_y.expand(C, 1, 3, 3)

    def get_edge_magnitude(img_batch):
        # Flatten channels to batch dimension to apply grayscale Sobel or handle per channel
        grad_x = F.conv2d(img_batch, sobel_x, groups=C, padding=1)
        grad_y = F.conv2d(img_batch, sobel_y, groups=C, padding=1)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return magnitude.mean(dim=1, keepdim=True) # Average channels to get single edge map

    # Compute edge maps for both blocks independently
    edges_X = get_edge_magnitude(X) # (N, 1, H, W)
    edges_Y = get_edge_magnitude(Y) # (M, 1, H, W)

    # Cross-compare via Mean Squared Error or Mean Absolute Error across all pairs
    # Expand shapes to (N, M, 1, H, W)
    edges_X_exp = edges_X.unsqueeze(1).expand(N, M, 1, H, W)
    edges_Y_exp = edges_Y.unsqueeze(0).expand(N, M, 1, H, W)

    # Pairwise L1 similarity metric mapped to [0, 1] bounds roughly
    edge_diff = torch.mean(torch.abs(edges_X_exp - edges_Y_exp), dim=[2, 3, 4])
    return 1.0 / (1.0 + edge_diff) # Return similarity score matrix (N, M)


def batch_lpips_distance(X: torch.Tensor, Y: torch.Tensor, lpips_model) -> torch.Tensor:
    """
    Computes pairwise LPIPS distances between batch X (N, C, H, W) 
    and batch Y (M, C, H, W) cleanly on the GPU.
    
    Returns a matrix of shape (N, M).
    """
    N, C, H, W = X.shape
    M = Y.shape[0]

    # 1. Scale inputs from [0, 1] to LPIPS expected [-1, 1] range
    X_scaled = X * 2.0 - 1.0
    Y_scaled = Y * 2.0 - 1.0

    # 2. Expand tensors to compute all N x M cross-combinations
    # Shapes become: (N * M, C, H, W)
    X_exp = X_scaled.unsqueeze(1).expand(N, M, C, H, W).reshape(N * M, C, H, W)
    Y_exp = Y_scaled.unsqueeze(0).expand(N, M, C, H, W).reshape(N * M, C, H, W)

    # 3. Stream through the pre-loaded model
    # LPIPS returns a distance tensor of shape (N * M, 1, 1, 1)
    with torch.no_grad():
        distances = lpips_model(X_exp, Y_exp)
        
    # 4. Flatten and reshape back to a beautiful (N, M) coordinate matrix
    return distances.reshape(N, M)

def batch_edge_similarity(batch_A, batch_B, model, method='ldc'):
    """
    Computes pairwise SSIM similarity between edge maps of two batches.
    """
    device = batch_A.device
    N, C, H, W = batch_A.shape
    M = batch_B.shape[0]

    def get_batch_edges(batch):
        if method == 'ldc':
            # LDC Preprocessing: Scale to 255 and subtract mean
            mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(device)
            inputs = (batch * 255.0) - mean
            outputs = model(inputs)
            edges = torch.sigmoid(outputs[-1]) # Use fused output
        elif method == 'hed':
            edges = model(batch)
        
        # Normalize each edge map in the batch to [0, 1]
        # Reshape to (B, -1) to find max per image
        b_size = edges.shape[0]
        max_vals = edges.view(b_size, -1).max(dim=1)[0].view(b_size, 1, 1, 1)
        return edges / (max_vals + 1e-7)

    # 1. Get edge maps for both batches once
    edges_A = get_batch_edges(batch_A) # (N, 1, H, W)
    edges_B = get_batch_edges(batch_B) # (M, 1, H, W)

    # 2. Compute Pairwise SSIM (Requires CPU/Numpy for skimage)
    # Note: For massive batches, you can use a PyTorch SSIM implementation 
    # to stay on GPU, but here we follow your skimage preference.
    eA_np = edges_A.squeeze(1).cpu().numpy()
    eB_np = edges_B.squeeze(1).cpu().numpy()
    
    results = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            results[i, j] = ssim(eA_np[i], eB_np[j], data_range=1.0)
            
    return torch.from_numpy(results)
    
def _pairwise_dists_chunked(feats_A, feats_B, dists_model, chunk_size=16):
    """
    Computes pairwise DISTS between two sets of 3-channel maps
    without materializing the full N*M batch at once.
    feats_A: [N, 3, H, W]
    feats_B: [M, 3, H, W]
    Returns: numpy array [N, M]
    """
    N, M = feats_A.shape[0], feats_B.shape[0]
    out = np.zeros((N, M), dtype=np.float32)

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        chunk_A = feats_A[i_start:i_end]  # [ci, 3, H, W]
        ci = chunk_A.shape[0]

        for j_start in range(0, M, chunk_size):
            j_end = min(j_start + chunk_size, M)
            chunk_B = feats_B[j_start:j_end]  # [cj, 3, H, W]
            cj = chunk_B.shape[0]

            # Only ci*cj pairs in memory at once
            a_exp = chunk_A.repeat_interleave(cj, dim=0)   # [ci*cj, 3, H, W]
            b_exp = chunk_B.repeat(ci, 1, 1, 1)             # [ci*cj, 3, H, W]

            with torch.no_grad():
                scores = dists_model(a_exp, b_exp)           # [ci*cj]

            out[i_start:i_end, j_start:j_end] = (
                scores.view(ci, cj).cpu().numpy()
            )

            # Free the pair batch immediately
            del a_exp, b_exp, scores

    return out


def _extract_depth_maps(batch, model_dict, mini_batch_size=8):
    """Run depth model in mini-batches to avoid OOM on large batches."""
    model = model_dict['model']
    processor = model_dict['processor']
    device = batch.device
    all_maps = []

    for start in range(0, batch.shape[0], mini_batch_size):
        chunk = batch[start:start + mini_batch_size]
        images_list = [torch.clamp(img, 0, 1) for img in chunk]

        with torch.no_grad():
            inputs = processor(images=images_list, return_tensors="pt").to(device)
            outputs = model(**inputs)

        maps = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(batch.shape[2], batch.shape[3]),
            mode="bicubic", align_corners=False
        )
        b = maps.shape[0]
        mins = maps.view(b, -1).min(1)[0].view(-1, 1, 1, 1)
        maxs = maps.view(b, -1).max(1)[0].view(-1, 1, 1, 1)
        all_maps.append((maps - mins) / (maxs - mins + 1e-8))

    return torch.cat(all_maps, dim=0)  # [N, 1, H, W]


def _extract_edges(batch, model, method='ldc', mini_batch_size=8):
    """Run edge model in mini-batches."""
    device = batch.device
    all_edges = []

    for start in range(0, batch.shape[0], mini_batch_size):
        chunk = batch[start:start + mini_batch_size]
        with torch.no_grad():
            if method == 'ldc':
                mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(device)
                out = model((chunk * 255.0) - mean)
                edges = torch.sigmoid(out[-1])
            else:
                edges = model(chunk)

        b = edges.shape[0]
        max_v = edges.view(b, -1).max(1)[0].view(-1, 1, 1, 1)
        all_edges.append(edges / (max_v + 1e-8))

    return torch.cat(all_edges, dim=0)  # [N, 1, H, W]


def batch_depth_analysis(batch_A, batch_B, model_dict, dists_model, dists_chunk=8, depth_mini_batch=8):

    depth_A = _extract_depth_maps(batch_A, model_dict, depth_mini_batch)
    depth_B = _extract_depth_maps(batch_B, model_dict, depth_mini_batch)
    N, M = depth_A.shape[0], depth_B.shape[0]

    # DISTS: chunked pairwise, no giant intermediate tensor
    dA3 = depth_A.repeat(1, 3, 1, 1)
    dB3 = depth_B.repeat(1, 3, 1, 1)
    dists_grid = _pairwise_dists_chunked(dA3, dB3, dists_model, chunk_size=dists_chunk)

    # CPU metrics unchanged
    dA_np = depth_A.squeeze(1).cpu().numpy()
    dB_np = depth_B.squeeze(1).cpu().numpy()
    mae_grid = np.zeros((N, M))
    ssim_grid = np.zeros((N, M))
    spear_grid = np.zeros((N, M))

    from scipy.stats import spearmanr
    for i in range(N):
        for j in range(M):
            mae_grid[i, j] = np.mean(np.abs(dA_np[i] - dB_np[j]))
            ssim_grid[i, j] = ssim(dA_np[i], dB_np[j], data_range=1.0)
            corr, _ = spearmanr(dA_np[i].flatten(), dB_np[j].flatten())
            spear_grid[i, j] = corr if not np.isnan(corr) else 0.0

    return mae_grid, ssim_grid, spear_grid, dists_grid


def batch_edge_analysis(batch_A, batch_B, model,  dists_model, method='ldc', dists_chunk=8, edge_mini_batch=8):

    edges_A = _extract_edges(batch_A, model, method, edge_mini_batch)
    edges_B = _extract_edges(batch_B, model, method, edge_mini_batch)
    N, M = edges_A.shape[0], edges_B.shape[0]

    eA3 = edges_A.repeat(1, 3, 1, 1)
    eB3 = edges_B.repeat(1, 3, 1, 1)
    dists_grid = _pairwise_dists_chunked(eA3, eB3, dists_model, chunk_size=dists_chunk)

    eA_np = edges_A.squeeze(1).cpu().numpy()
    eB_np = edges_B.squeeze(1).cpu().numpy()
    ssim_grid = np.zeros((N, M))
    fom_grid  = np.zeros((N, M))
    haus_grid = np.zeros((N, M))

    from scipy.ndimage import distance_transform_edt
    from scipy.spatial.distance import directed_hausdorff

    for i in range(N):
        bin_A = eA_np[i] > 0.1
        pts_A = np.argwhere(bin_A)
        dist_trans_A = distance_transform_edt(~bin_A)

        for j in range(M):
            ssim_grid[i, j] = ssim(eA_np[i], eB_np[j], data_range=1.0)
            bin_B = eB_np[j] > 0.1
            pts_B = np.argwhere(bin_B)

            if np.any(bin_A) and np.any(bin_B):
                d_i = dist_trans_A[bin_B]
                fom_grid[i, j] = (
                    np.sum(1.0 / (1.0 + (1.0/9.0) * (d_i ** 2)))
                    / max(np.sum(bin_A), np.sum(bin_B))
                )
                h1 = directed_hausdorff(pts_A, pts_B)[0]
                h2 = directed_hausdorff(pts_B, pts_A)[0]
                haus_grid[i, j] = max(h1, h2)
            else:
                haus_grid[i, j] = 1000.0

    return ssim_grid, dists_grid, fom_grid, haus_grid
