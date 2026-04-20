import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from experiments.reference_methods.style_transfer.photorealistic.deeppreset.src.utils import *
from experiments.reference_methods.style_transfer.photorealistic.deeppreset.src.networks.network import get_model, PRESET_PREDICTION_IMG_SIZE

class ToTensor(object):
    def __call__(self, tmp):
        tmp = tmp / 255.0
        tmp = (tmp - 0.5)/0.5
        tmp = tmp.transpose((2, 0, 1))
        return torch.from_numpy(tmp).unsqueeze(0).float()

class DeepPreset(object):
    def __init__(self, args):
        self.device = getattr(args, 'device', 'cuda')
        ckpt = torch.load(args.ckpt, map_location=self.device, weights_only=False)
        
        # Load model
        # We need to make sure ckpt['opts'] is compatible or we might need to mock it if it's a Namespace
        # Usually torch.load preserves the object type.
        
        self.G = get_model(ckpt['opts'].g_net)(ckpt['opts']).to(self.device)
        self.G.load_state_dict(ckpt['G'])
        self.G.eval()
        
        self.totensor = ToTensor()
        self.preset_handler = PresetHandler()
        self.img_size = size_str2tuple(args.size) if hasattr(args, 'size') and args.size else PRESET_PREDICTION_IMG_SIZE
        self.p_only = getattr(args, 'p_only', False)

    def process(self, content_tensor, style_tensor):
        # content_tensor, style_tensor: [C, H, W] in [0, 1] (or whatever the caller provides)
        # The original code takes PIL images, resizes them, converts to numpy, then ToTensor.
        # ToTensor does: / 255.0, (x - 0.5)/0.5, transpose.
        
        # If we receive tensors in [0, 1], we need to normalize to [-1, 1].
        # And ensure they are [1, C, H, W]
        
        if content_tensor.dim() == 3:
            content_tensor = content_tensor.unsqueeze(0)
        if style_tensor.dim() == 3:
            style_tensor = style_tensor.unsqueeze(0)
            
        # Normalize [0, 1] -> [-1, 1]
        content = (content_tensor - 0.5) / 0.5
        style = (style_tensor - 0.5) / 0.5
        
        content = content.to(self.device)
        style = style.to(self.device)
        
        # Resize if needed? The original code resizes to self.img_size (352x352) or style size.
        # "pil_cont = pil_cont.resize(self.img_size, resample=Image.BICUBIC)"
        # We should probably respect the input size or resize to native size.
        # The method wrapper will handle resizing if needed.
        # But the model might expect specific size (multiples of 32?).
        # The network has depth (g_depth), so downsampling happens.
        
        with torch.no_grad():
            img_out, preset_out, preset_emb = self.G.stylize(content, style, 
                                                           self.img_size == PRESET_PREDICTION_IMG_SIZE, 
                                                           preset_only=self.p_only)
        
        # Denormalize [-1, 1] -> [0, 1]
        img_out = (img_out + 1) / 2
        img_out = torch.clamp(img_out, 0, 1)
        
        return img_out
