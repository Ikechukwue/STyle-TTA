import torch

def is_torch2_available():
    return hasattr(torch.nn.functional, 'scaled_dot_product_attention')

def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None
    return generator
