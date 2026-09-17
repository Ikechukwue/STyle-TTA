import torch
from dataclasses import dataclass
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
import lpips

@dataclass
class MetricResult:
    value: float

# Cache LPIPS network globally to avoid reloading weights every call
_LPIPS_NET = None

def _get_lpips_net(device):
    global _LPIPS_NET
    if _LPIPS_NET is None:
        _LPIPS_NET = lpips.LPIPS(net='alex').to(device).eval()
    return _LPIPS_NET

def compute_ssim(outputs: torch.Tensor, references: torch.Tensor) -> MetricResult:
    """Computes standard SSIM."""
    # Ensure range [0, 1] for torchmetrics
    if outputs.max() > 1.0: outputs = outputs / 255.0
    if references.max() > 1.0: references = references / 255.0
    
    score = ssim_fn(outputs, references, data_range=1.0).item()
    return MetricResult(value=score)

def compute_luminance_ssim(outputs: torch.Tensor, references: torch.Tensor) -> MetricResult:
    """Converts images to grayscale (luminance) first, then computes SSIM."""
    if outputs.max() > 1.0: outputs = outputs / 255.0
    if references.max() > 1.0: references = references / 255.0
    
    # Grayscale conversion weights: Y = 0.299R + 0.587G + 0.114B
    w = torch.tensor([0.299, 0.587, 0.114], device=outputs.device).view(1, 3, 1, 1)
    out_gray = torch.sum(outputs * w, dim=1, keepdim=True)
    ref_gray = torch.sum(references * w, dim=1, keepdim=True)
    
    score = ssim_fn(out_gray, ref_gray, data_range=1.0).item()
    return MetricResult(value=score)

def compute_lpips(outputs: torch.Tensor, references: torch.Tensor) -> MetricResult:
    """Computes Learned Perceptual Image Patch Similarity."""
    # LPIPS expects inputs in range [-1, 1]
    if outputs.max() <= 1.0:
        out_norm = (outputs * 2.0) - 1.0
        ref_norm = (references * 2.0) - 1.0
    else:
        out_norm = (outputs / 127.5) - 1.0
        ref_norm = (references / 127.5) - 1.0
        
    loss_fn = _get_lpips_net(outputs.device)
    with torch.no_grad():
        score = loss_fn(out_norm, ref_norm).mean().item()
    return MetricResult(value=score)
