import numpy as np
from skimage.metrics import structural_similarity as ssim

a = np.random.rand(256, 256).astype(np.float32)
b = np.random.rand(256, 256).astype(np.float32)

# What does your current ssim import give?
print(ssim(a, b, data_range=1.0, channel_axis=-1))  
print(ssim(a,a, data_range=1.0, channel_axis=-1))
# Should be near 0, definitely in [-1, 1]
# If it prints something like 130.0, that's your problem
