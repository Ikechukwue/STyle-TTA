import torch
import numpy as np
from experiments.metrics.color_metrics import (_get_fid_model, _get_vgg_model, _extract_vgg_features, _get_gram_matrix)
from experiments.data import create_dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse

@torch.no_grad()
def extract_evaluation_features(images: torch.Tensor):
    """
    Extracts both FID activations and VGG Gram matrices from a single batch.
    """
    # 1. FID (Inception)
    fid_model = _get_fid_model()
    from torchvision.transforms import functional as F
    fid_input = torch.stack([F.resize(img, [299, 299], antialias=True) * 2 - 1 for img in images])
    # Ensure input is on same device as model
    fid_input = fid_input.to(next(fid_model.parameters()).device)
    fid_activations = fid_model(fid_input).cpu().numpy()

    # 2. VGG Grams
    vgg_model = _get_vgg_model()
    layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    vgg_input = images.to(next(vgg_model.parameters()).device)
    vgg_features = _extract_vgg_features(vgg_input, vgg_model, layers)
    
    # Calculate grams for this batch and keep on CPU
    batch_grams = {
        layer: _get_gram_matrix(feat).cpu() 
        for layer, feat in vgg_features.items()
    }

    return fid_activations, batch_grams

def cache_features(args):
    output_path = Path(args.output_dir) / f"features_{args.dataset}_{args.split}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(dataset_name=args.dataset, 
                             data_path=args.data_path, 
                             split=args.split)
    
    # Batch size 16 is safe for A100/V100
    test_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=False, num_workers=4)

    # Accumulators
    all_fid_feats = []
    # Initialize dictionary of lists for each layer
    layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    all_vgg_grams = {layer: [] for layer in layers}

    # Use the GPU
    for batch in tqdm(test_loader, desc=f"Extracting {args.split}"):
        # Assuming your dataset returns (image, label) or (image, path)
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch

        fid_batch, grams_batch = extract_evaluation_features(imgs)
        
        all_fid_feats.append(fid_batch)
        for layer in layers:
            all_vgg_grams[layer].append(grams_batch[layer])

    # Final Merge
    results = {
        'fid_activations': np.concatenate(all_fid_feats, axis=0),
        'vgg_gram_matrices': {
            layer: torch.cat(all_vgg_grams[layer], dim=0) 
            for layer in layers
        }
    }

    print(f"Saving features to {output_path}...")
    torch.save(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./feature_cache")
    args = parser.parse_args()
    
    cache_features(args)
