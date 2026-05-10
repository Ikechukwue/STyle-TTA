from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
    TASK_TYPE,
    DATASET_SPLITS,
)
from experiments.utils.reproducibility import random_seed, worker_seed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Generator
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from accelerate import Accelerator
from tqdm import tqdm
import timm
from pathlib import Path
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio

def debug_dataset_structure(dataset, depth=0):
    indent = "  " * depth
    print(f"{indent}🔍 Level {depth}: {type(dataset).__name__}")
    labels = getattr(dataset, 'targets', None)
    if labels is not None:
        print(f"Found targets at level {depth}")
        return
    # Check if this is a wrapper (like Subset or ConcatDataset)
    if hasattr(dataset, 'dataset'):
        debug_dataset_structure(dataset.dataset, depth + 1)
    
    # Once we've identified the chain, let's look at the actual data
    if depth == 0:
        print("\n--- Sample Inspection ---")
        try:
            sample = dataset[0]
            print(f"Full sample type: {type(sample)}")
            labels = getattr(dataset, 'targets', None)
            if labels is not None:
                print(f"Found targets at level {depth}")
                return
            if isinstance(sample, (tuple, list)):
                print(f"Sample length: {len(sample)}")
                for i, item in enumerate(sample):
                    _inspect_item(item, f"Element [{i}]")
            elif isinstance(sample, dict):
                for key, value in sample.items():
                    _inspect_item(value, f"Key '{key}'")
            else:
                _inspect_item(sample, "Single item")
        except Exception as e:
            print(f"❌ Could not retrieve sample: {e}")

def _inspect_item(item, label):
    if hasattr(item, 'shape'):
        info = f"Shape: {list(item.shape)}"
    elif hasattr(item, '__len__') and not isinstance(item, str):
        info = f"Length: {len(item)}"
    else:
        info = f"Value: {item}"
    print(f"  👉 {label} | Type: {type(item).__name__} | {info}")

if __name__=="__main__":
    dataset = "imagenet"
    split = "train@test_r"
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=256),
    ])
    set_A = create_dataset(dataset, "./data", split, transform)
    debug_dataset_structure(set_A)
    # Drill down to the ImageFolder layer
    base_ds = set_A.dataset.dataset 

    mapping = base_ds.class_to_idx


