import torch
from dataclasses import dataclass
from torchmetrics.image.fid import FrechetInceptionDistance

@dataclass
class MetricResult:
    value: float

def compute_fid_tensor(outputs: torch.Tensor, references: torch.Tensor) -> MetricResult:
    """
    Computes FID between output and reference image batches.
    Expects tensors shape (N, C, H, W), type float32, range [0, 1] or [0, 255].
    """
    device = outputs.device
    # torchmetrics FID requires torch.uint8 [0, 255]
    if outputs.max() <= 1.0:
        outputs = (outputs * 255).to(torch.uint8)
    else:
        outputs = outputs.to(torch.uint8)
        
    if references.max() <= 1.0:
        references = (references * 255).to(torch.uint8)
    else:
        references = references.to(torch.uint8)

    # Initialize FID metric (using standard inception features)
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
    
    # Update real and fake images
    fid.update(references, real=True)
    fid.update(outputs, real=False)
    
    score = fid.compute().item()
    return MetricResult(value=score)
