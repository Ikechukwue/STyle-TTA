import torch
from dataclasses import dataclass
from code.experiments.metrics.distribution_metrics import compute_fid_tensor
from code.experiments.metrics.feature_metrics import compute_lpips

@dataclass
class MetricResult:
    value: float

def compute_artfid_tensor(outputs: torch.Tensor, references: torch.Tensor, alpha: float = 1.0) -> MetricResult:
    """
    Computes ArtFID = FID(outputs, references) + alpha * Mean(LPIPS(outputs, references))
    """
    fid_score = compute_fid_tensor(outputs, references).value
    lpips_score = compute_lpips(outputs, references).value
    
    artfid_score = fid_score + (alpha * lpips_score)
    return MetricResult(value=artfid_score)
