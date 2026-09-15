import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import v2

from config.helpers import get_class_name, get_names
from experiments.data import create_dataset
from experiments.tta.reference_db_setup import build_reference_db

roster_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(224, antialias=True),
        v2.CenterCrop((224, 224)),
    ]
)


def crop_to_content_aspect_ratio(stylized_img, content_img):
    cw, ch = content_img.size
    sw, sh = stylized_img.size

    content_ar = cw / ch

    if content_ar > 1.0:
        target_w = sw
        target_h = int(round(sw / content_ar))
    else:
        target_h = sh
        target_w = int(round(sh * content_ar))

    left = (sw - target_w) // 2
    top = (sh - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return stylized_img.crop((left, top, right, bottom))


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


def load_image_safe(path, fallback_size=(224, 224)):
    p = Path(path)
    if p.exists():
        return Image.open(p).convert("RGB")
    return Image.new("RGB", fallback_size, color=(200, 200, 200))


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


def generate_dataset_3x6_roster(
    dataset_name: str,
    split: str,
    mapping_path: str,
    stylized_folder: str,
    sample_indices: list[int],
    data_path: str = "./data",
    cache_base_dir: str = "/home/stud/nemmler/retristyle/data/augmented_cache",
    view_name: str = "view_001.png",
    input_size: int = 224,
    seed: int = 71397589,
    output_path: str = "output/images/dataset_roster_3x6.png",
):
    assert len(sample_indices) == 6

    raw_test_set = create_dataset(
        dataset_name=dataset_name, data_path=data_path, split=split, transform=None
    )
    test_set = create_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        split=split,
        transform=roster_transform,
    )
    ref_db = build_reference_db(
        dataset=dataset_name,
        data_path=data_path,
        input_size=input_size,
        seed=seed,
        split=split if split != "test" else "train",
        show=False,
    )

    with open(mapping_path, "r") as f:
        retrieval_mapping = json.load(f)

    fig, axes = plt.subplots(3, 6, figsize=(6 * 3.2, 3 * 3.2))
    row_labels = ["Style", "Content", "Stylized"]

    for col_idx, sample_idx in enumerate(sample_indices):
        sample_str_idx = str(sample_idx)

        # Row 0: Style Image
        style_ref_idx = retrieval_mapping[sample_str_idx][0][0]
        style_img_tensor = ref_db.get_image(style_ref_idx)
        style_img = load_and_standardize(
            style_img_tensor, target_size=(input_size, input_size)
        )

        ax_style = axes[0, col_idx]
        ax_style.imshow(style_img)
        ax_style.axis("off")

        # Row 1: Content Image
        content_item = test_set[sample_idx]
        content_img = (
            content_item[0]
            if isinstance(content_item, (tuple, list))
            else content_item
        )
        if not isinstance(content_img, Image.Image):
            content_img = tensor_to_pil(content_img)
        content_img = load_and_standardize(
            content_img, target_size=(input_size, input_size)
        )

        ax_content = axes[1, col_idx]
        ax_content.imshow(content_img)
        ax_content.axis("off")

        # Row 2: Stylized Image
        folder_idx = f"{sample_idx:05d}"
        stylized_path = (
            Path(cache_base_dir) / stylized_folder / folder_idx / view_name
        )

        raw_content_item = raw_test_set[sample_idx]
        raw_content_img = (
            raw_content_item[0]
            if isinstance(raw_content_item, (tuple, list))
            else raw_content_item
        )
        if not isinstance(raw_content_img, Image.Image):
            raw_content_img = tensor_to_pil(raw_content_img)

        raw_stylized = load_image_safe(stylized_path)
        cropped_stylized = crop_to_content_aspect_ratio(
            raw_stylized, raw_content_img
        )
        final_stylized = load_and_standardize(
            cropped_stylized, target_size=(input_size, input_size)
        )

        ax_stylized = axes[2, col_idx]
        ax_stylized.imshow(final_stylized)
        ax_stylized.axis("off")

    for row_idx in range(3):
        axes[row_idx, 0].text(
            -0.1,
            0.5,
            row_labels[row_idx],
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=90,
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
    plt.close()
    print(f"Roster saved to {output_path}")


if __name__ == "__main__":
    mapping_template = "/home/stud/nemmler/retristyle/results/retrieval_mapping/{ds}/retrieval_mapping_dino_test_r_s71397589.json"

    dataset_configs = [
        {
            "name": "imagenet",
            "split": "test_r",
            "mapping_path": mapping_template.format(ds="imagenet", split="test_r"),
            "stylized_folder": "dino_imagenet_test_r_s71397589",
            "sample_indices": [29680, 3181, 26889, 25190, 100, 500],
            "output_path": "output/images/imagenet/roster_3x6.png",
        },
        {
            "name": "eurosat",
            "split": "ucmerced",
            "mapping_path": mapping_template.format(ds="eurosat", split="ucmerced"),
            "stylized_folder": "dino_eurosat_ucmerced_s71397589",
            "sample_indices": [10, 20, 30, 40, 50, 60],
            "output_path": "output/images/eurosat/roster_3x6.png",
        },
        {
            "name": "midog",
            "split": "test",
            "mapping_path": mapping_template.format(ds="midog", split="test"),
            "stylized_folder": "dino_midog_test_s71397589",
            "sample_indices": [100, 101, 110, 103, 104, 105],
            "output_path": "output/images/midog/roster_3x6.png",
        },
        {
            "name": "epistr",
            "split": "test",
            "mapping_path": mapping_template.format(ds="epistr", split="test"),
            "stylized_folder": "dino_epistr_test_s71397589",
            "sample_indices": [100, 101, 110, 103, 104, 105],
            "output_path": "output/images/epistr/roster_3x6.png",
        },
        {
            "name": "camelyon17wilds",
            "split": "test",
            "mapping_path": mapping_template.format(ds="camelyon17wilds", split="test"),
            "stylized_folder": "dino_camelyon17wilds_test_s71397589",
            "sample_indices": [50, 51, 52, 53, 54, 55],
            "output_path": "output/images/camelyon17wilds/roster_3x6.png",
        },
    ]

    for config in dataset_configs:
        print(f"Processing dataset: {config['name']}...")
        generate_dataset_3x6_roster(
            dataset_name=config["name"],
            split=config["split"],
            mapping_path=config["mapping_path"],
            stylized_folder=config["stylized_folder"],
            sample_indices=config["sample_indices"],
            output_path=config["output_path"],
        )
