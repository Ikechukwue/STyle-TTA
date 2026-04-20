import torch
# Load a small piece of the data
checkpoint = torch.load("./data/feature_cache/ViT-B-16/train.pt", map_location='cpu')
labels = checkpoint["labels"]

print(f"Type: {type(labels)}")
print(f"Shape: {labels.shape}")
print(f"First 5 values: {labels[:5]}")
