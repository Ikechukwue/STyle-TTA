import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import v2
from code.config.constants import ALL_SPLITS, ALL_DATASETS
from code.config.helpers import get_class_name, get_names
from code.experiments.data import create_dataset
from code.experiments.tta.reference_db_setup import build_reference_db


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


def generate_roster(
    datasets,
    data_path="./data",
    input_size=224,
    seed=71397589,
    output_path="output/images/dataset_roster.png",
):
    num_datasets = len(datasets)
    class_list = get_names()

    fig, axes = plt.subplots(
        2, num_datasets, figsize=(num_datasets * 3.5, 2 * 3.5)
    )

    if num_datasets == 1:
        axes = axes.reshape(2, 1)

    row_titles = ["In-Domain (Style)", "Out-of-Domain (Content)"]

    for col_idx, ds_info in enumerate(datasets):
        ds_name = ds_info["name"]
        split = ds_info.get("split", "test")
        sample_idx = ds_info.get("sample_idx", 0)
        mapping_path = ds_info["mapping_path"]

        if ds_name == "epistr":
            style_img_path = "/home/stud/nemmler/retristyle/data/epistr/NKI/train/epi/epi19.jpg"
            content_img_path = "/home/stud/nemmler/retristyle/data/epistr/IHC/test/epi/epi8.png"

            style_img = Image.open(style_img_path).convert("RGB")
            style_img = load_and_standardize(
                style_img, target_size=(input_size, input_size)
            )

            content_img = Image.open(content_img_path).convert("RGB")
            content_img = load_and_standardize(
                content_img, target_size=(input_size, input_size)
            )
            content_class_name = "Epithelium"
        else:
            ood_dataset = create_dataset(
                dataset_name=ds_name,
                data_path=data_path,
                split=split,
                transform=None,
            )

            ref_db = build_reference_db(
                dataset=ds_name,
                data_path=data_path,
                input_size=input_size,
                seed=seed,
                split=split if split != "test" else "train",
                show=False,
            )

            with open(mapping_path, "r") as f:
                retrieval_mapping = json.load(f)

            sample_str_idx = str(sample_idx)

            # 1. Style Image
            style_ref_idx = retrieval_mapping[sample_str_idx][0][0]
            style_img_tensor = ref_db.get_image(style_ref_idx)
            style_img = load_and_standardize(
                style_img_tensor, target_size=(input_size, input_size)
            )

            # 2. Content Image
            content_item = ood_dataset[sample_idx]
            content_img = (
                content_item[0]
                if isinstance(content_item, (tuple, list))
                else content_item
            )
            content_label = (
                content_item[1]
                if isinstance(content_item, (tuple, list))
                else None
            )

            if not isinstance(content_img, Image.Image):
                content_img = tensor_to_pil(content_img)

            content_img = load_and_standardize(
                content_img, target_size=(input_size, input_size)
            )
            content_class_name = (
                get_class_name(content_label, class_list)
                if content_label is not None
                else "N/A"
            )

        # Plot Style (ID) Image using ALL_SPLITS
        ax_id = axes[0, col_idx]
        ax_id.imshow(style_img)
        ax_id.axis("off")
        ax_id.text(
            0.5,
            -0.05,
            f"Dataset: {ALL_DATASETS[ds_name]}",
            transform=ax_id.transAxes,
            ha="center",
            va="top",
            fontsize=15,
        )

        # Plot Content (OOD) Image
        ax_ood = axes[1, col_idx]
        ax_ood.imshow(content_img)
        ax_ood.axis("off")
        ax_ood.text(
            0.5,
            -0.05,
            f"Dataset: {ALL_SPLITS[ds_name]}",
            transform=ax_ood.transAxes,
            ha="center",
            va="top",
            fontsize=15,
        )

    for row_idx in range(2):
        axes[row_idx, 0].text(
            -0.1,
            0.5,
            row_titles[row_idx],
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=15,
            fontweight="bold",
            rotation=90,
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.2, dpi=300)
    print(f"Roster saved to {output_path}")


if __name__ == "__main__":
    path = "/home/stud/nemmler/retristyle/results/retrieval_mapping/{ds}/retrieval_mapping_dino_test_r_s71397589.json"
    datasets = [
        {
            "name": "imagenet",
            "split": "test_r",
            "sample_idx": 100,
            "mapping_path": path.format(ds="imagenet"),
        },
        {
            "name": "eurosat",
            "split": "ucmerced",
            "sample_idx": 111,
            "mapping_path": path.format(ds="eurosat"),
        },
        {
            "name": "midog",
            "split": "test",
            "sample_idx": 500,
            "mapping_path": path.format(ds="midog"),
        },
        {
            "name": "camelyon17wilds",
            "split": "test",
            "sample_idx": 99,
            "mapping_path": path.format(ds="camelyon17wilds"),
        },
        {
            "name": "epistr",
            "split": "test",
            "sample_idx": 100,
            "mapping_path": path.format(ds="epistr"),
        },
    ]

    generate_roster(
        datasets=datasets,
        data_path="./data",
        input_size=224,
        seed=71397589,
        output_path="output/images/dataset_roster.png",
    )
