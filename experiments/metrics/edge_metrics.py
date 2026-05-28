import torch
from dataclasses import dataclass
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

@dataclass
class MetricResult:
    value: float

def _get_edges(tensor: torch.Tensor) -> torch.Tensor:
    """Extracts edge maps using standard Sobel filters via 2D convolution."""
    if tensor.shape[1] == 3:  # Convert to grayscale if RGB
        w = torch.tensor([0.299, 0.587, 0.114], device=tensor.device).view(1, 3, 1, 1)
        tensor = torch.sum(tensor * w, dim=1, keepdim=True)
        
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=tensor.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(tensor, sobel_x, padding=1)
    grad_y = F.conv2d(tensor, sobel_y, padding=1)
    
    return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

def compute_edge_ssim(outputs: torch.Tensor, references: torch.Tensor) -> MetricResult:
    edge_out = _get_edges(outputs)
    edge_ref = _get_edges(references)
    # Normalize back to [0,1] baseline dynamic range
    edge_out = torch.clamp(edge_out / edge_out.max().clamp(min=1e-5), 0, 1)
    edge_ref = torch.clamp(edge_ref / edge_ref.max().clamp(min=1e-5), 0, 1)
    
    score = ssim_fn(edge_out, edge_ref, data_range=1.0).item()
    return MetricResult(value=score)

def compute_edge_dists(outputs: torch.Tensor, references: torch.Tensor) -> MetricResult:
    """Mean Squared Error (MSE) between edge maps."""
    edge_out = _get_edges(outputs)
    edge_ref = _get_edges(references)
    score = F.mse_loss(edge_out, edge_ref).item()
    return MetricResult(value=score)

def compute_edge_adists(outputs: torch.Tensor, references: torch.Tensor) -> MetricResult:
    """Mean Absolute Error (MAE / L1) between edge maps."""
    edge_out = _get_edges(outputs)
    edge_ref = _get_edges(references)
    score = F.l1_loss(edge_out, edge_ref).item()
    return MetricResult(value=score)
