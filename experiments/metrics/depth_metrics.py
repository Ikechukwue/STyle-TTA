import torch
from dataclasses import dataclass
import torch.nn.functional as F
from scipy.stats import spearmanr

@dataclass
class MetricResult:
    value: float

_DEPTH_MODELS = {}

def _get_depth_map(tensor: torch.Tensor, method: str) -> torch.Tensor:
    """Extracts depth maps using MiDaS architectures depending on method string."""
    global _DEPTH_MODELS
    device = tensor.device
    
    if method not in _DEPTH_MODELS:
        # Map method tags to torchvision/torch.hub variants
        model_type = "MiDaS_small" if "small" in method or "midas" in method else "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        _DEPTH_MODELS[method] = midas.to(device).eval()
        
    model = _DEPTH_MODELS[method]
    
    # Handle RGB normalization expectations for depth estimators
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
        
    # Resize to standard depth input sizes if needed (e.g., 384x384)
    orig_shape = tensor.shape[2:]
    input_resized = F.interpolate(tensor, size=(384, 384), mode="bicubic", align_corners=False)
    
    with torch.no_grad():
        prediction = model(input_resized)
        depth = F.interpolate(prediction.unsqueeze(1), size=orig_shape, mode="bicubic", align_corners=False)
        
    return depth

def compute_depth_spearman(outputs: torch.Tensor, references: torch.Tensor, method: str) -> MetricResult:
    """Computes spatial Spearman rank correlation between depth layouts."""
    depth_out = _get_depth_map(outputs, method).cpu().numpy().flatten()
    depth_ref = _get_depth_map(references, method).cpu().numpy().flatten()
    
    # Compute rank correlation using scipy
    score, _ = spearmanr(depth_out, depth_ref)
    return MetricResult(value=float(score))

def compute_depth_dists(outputs: torch.Tensor, references: torch.Tensor, method: str) -> MetricResult:
    """MSE distance between normalized depth maps."""
    depth_out = _get_depth_map(outputs, method)
    depth_ref = _get_depth_map(references, method)
    
    # Range-normalize depth to align comparisons
    depth_out = (depth_out - depth_out.min()) / (depth_out.max() - depth_out.min() + 1e-5)
    depth_ref = (depth_ref - depth_ref.min()) / (depth_ref.max() - depth_ref.min() + 1e-5)
    
    score = F.mse_loss(depth_out, depth_ref).item()
    return MetricResult(value=score)

def compute_depth_adists(outputs: torch.Tensor, references: torch.Tensor, method: str) -> MetricResult:
    """L1 / Absolute distance between normalized depth maps."""
    depth_out = _get_depth_map(outputs, method)
    depth_ref = _get_depth_map(references, method)
    
    depth_out = (depth_out - depth_out.min()) / (depth_out.max() - depth_out.min() + 1e-5)
    depth_ref = (depth_ref - depth_ref.min()) / (depth_ref.max() - depth_ref.min() + 1e-5)
    
    score = F.l1_loss(depth_out, depth_ref).item()
    return MetricResult(value=score)
