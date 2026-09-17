import torch
data = torch.load("./data/feature_cache/ViT-B-16/train.pt", weights_only=True)
print("features shape:", data['features'].shape)
print("labels shape:", data['labels'].shape)
