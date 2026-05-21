# debug_method.py
import torch
from experiments.reference_methods.style_transfer_factory import create_color_transfer_method

device = "cuda" if torch.cuda.is_available() else "cpu"
method_name = "sanet"
weights = "./data/models/style_transfer/sanet.pth"

print(f"Testing {method_name}...")

# method is the wrapper, network is the actual nn.Module
method, network = create_color_transfer_method(method_name, weights)

# 1. Set the internal network to eval mode
if network is not None:
    network.to(device)
    network.eval()
elif hasattr(method, 'model'): # Some wrappers store it in .model
    method.model.to(device)
    method.model.eval()

# 2. Dummy data
c = torch.randn(1, 3, 256, 256).to(device)
s = torch.randn(1, 3, 256, 256).to(device)

# 3. Test the forward pass
with torch.no_grad():
    try:
        # Most wrappers implement __call__ or a forward-like method
        out = method(c, s) 
        print(f"Output shape: {out.shape}")
        print(f"Output stats: Mean={out.mean().item():.4f}, Max={out.max().item():.4f}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
