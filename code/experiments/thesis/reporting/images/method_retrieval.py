import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import v2
from code.config.helpers import load_json
from code.experiments.data import create_dataset
from code.experiments.tta.reference_db_setup import build_reference_db

def load_and_standardize(img_input, target_size=(224, 224)):
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(target_size, antialias=True),
        v2.CenterCrop(target_size),
    ])
    tensor = transform(img_input)
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return (img * 255).astype("uint8")

def plot_top_k_retrieved(
    dataset_name="imagenet",
    split="test_r",
    data_path="./data",
    sample_idx=0,
    k=5,
    mapping_path="",
    output_path="retrieved_top_k.png",
):
    test_set = create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split=split,
        transform=None,
    )
    ref_db = build_reference_db(
        dataset=dataset_name,
        data_path=data_path,
        input_size=224,
        seed=71397589,
        split=split,
        show=False,
    )

    with open(mapping_path, "r") as f:
        retrieval_mapping = json.load(f)

    content_item = test_set[sample_idx]
    content_img = content_item[0] if isinstance(content_item, (tuple, list)) else content_item
    content_std = load_and_standardize(content_img)

    retrieved_refs = retrieval_mapping[str(sample_idx)][:k]

    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))

    axes[0].imshow(content_std)
    axes[0].set_title("Content Image", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    for i, ref in enumerate(retrieved_refs):
        ref_idx = ref[0]
        style_img_tensor = ref_db.get_image(ref_idx)
        style_std = load_and_standardize(style_img_tensor)

        ax = axes[i + 1]
        ax.imshow(style_std)
        ax.set_title(f"Rank {i + 1}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

def main():
    dataset = "imagenet"
    split = "test_r"
    data_path = "./data"
    sample_idx = 29680
    k = 5
    mapping_path = "/home/stud/nemmler/retristyle/results/retrieval_mapping/imagenet/retrieval_mapping_dino_test_r_s71397589.json"
    output_path = "top_5_retrieved.png"

    plot_top_k_retrieved(
        dataset_name=dataset,
        split=split,
        data_path=data_path,
        sample_idx=sample_idx,
        k=k,
        mapping_path=mapping_path,
        output_path=output_path,
    )

if __name__ == "__main__":
    main()
