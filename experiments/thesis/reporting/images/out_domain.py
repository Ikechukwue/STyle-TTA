import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
from config.constants import ALL_SPLITS, TRUE_SPLITS
from config.helpers import get_names, get_class_name
from experiments.data import create_dataset


def tensor_to_pil(tensor):
    if isinstance(tensor, torch.Tensor):
        img = tensor.detach().cpu()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            if img.min() < 0 or img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            img = img.permute(1, 2, 0)
        img = (img.numpy() * 255).astype("uint8")
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(-1)
        return Image.fromarray(img)
    return tensor


def load_and_standardize(img_input, target_size=(224, 224)):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(min(target_size), antialias=True),
            v2.CenterCrop(target_size),
        ]
    )
    tensor = transform(img_input)
    return tensor_to_pil(tensor)


def extract_image_and_target(item):
    if isinstance(item, (tuple, list)):
        return item[0], item[1]
    return item, getattr(item, "target", None)


def build_class_index_map(dataset):
    class_to_indices = {}
    for idx in range(len(dataset)):
        _, target = extract_image_and_target(dataset[idx])
        if target is not None:
            target = int(target)
            if target not in class_to_indices:
                class_to_indices[target] = []
            class_to_indices[target].append(idx)
    return class_to_indices


def main():
    direct_ds = ["midog", "epistr", "camelyon17wilds"]
    dataset = "imagenet"
    split = TRUE_SPLITS[ALL_SPLITS[dataset]]
    data_path = "./data"
    target_class = 1
    output_path = f"output/images/domain_comparison_2x2_{}.png"
    seed = 42 

    np.random.seed(seed)

    in_domain_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split="train" if not dataset in direct_ds else f"train@{split}",
        transform=None,
    )

    out_domain_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split=split,
        transform=None,
    )

    in_map = build_class_index_map(in_domain_set)
    out_map = build_class_index_map(out_domain_set)

    if target_class not in in_map or len(in_map[target_class]) < 2:
        raise ValueError(
            f"Class {target_class} has fewer than 2 samples in in-domain dataset."
        )

    if target_class not in out_map or len(out_map[target_class]) < 2:
        raise ValueError(
            f"Class {target_class} has fewer than 2 samples in out-of-domain dataset."
        )

    in_indices = np.random.choice(in_map[target_class], size=2, replace=False)
    out_indices = np.random.choice(out_map[target_class], size=2, replace=False)

    class_list = get_names()
    class_name = get_class_name(target_class, class_list)

    grid = [
        [in_domain_set[in_indices[0]], out_domain_set[out_indices[0]]],
        [in_domain_set[in_indices[1]], out_domain_set[out_indices[1]]],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    col_titles = ["In-Domain", "Out-of-Domain"]

    for row in range(2):
        for col in range(2):
            ax = axes[row, col]
            raw_img, _ = extract_image_and_target(grid[row][col])
            processed_img = load_and_standardize(raw_img, target_size=(224, 224))
            ax.imshow(processed_img)
            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold", pad=10)

    fig.suptitle(f"Class: {class_name} ({target_class})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
