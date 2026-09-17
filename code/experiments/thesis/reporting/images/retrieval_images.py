import json
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image

from code.config.helpers import get_top_k
from code.experiments.data import create_dataset
from code.experiments.tta.reference_db_setup import build_reference_db

roster_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(224, antialias=True),
        v2.CenterCrop((224, 224)),
    ]
)


def get_comparison_indices(
    path_a: str | Path,
    path_b: str | Path,
    num_samples: int = 3,
    mode: str = "all_correct",
    seed: int | None = None,
    k: int = 1,
    max_idx: int = 100,  # Limit dataset search space
) -> list[int]:
    with open(path_a, "r") as f:
        data_a = json.load(f)
    with open(path_b, "r") as f:
        data_b = json.load(f)

    indices_a, _ = get_top_k(data_a, k=k)
    indices_b, _ = get_top_k(data_b, k=k)

    y_true_a = np.array([s["y_true"] for s in data_a["predictions"]])
    y_true_b = np.array([s["y_true"] for s in data_b["predictions"]])

    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError("path_a and path_b contain different y_true values.")

    correct_a = np.any(indices_a == y_true_a[:, None], axis=-1)
    correct_b = np.any(indices_b == y_true_b[:, None], axis=-1)

    matching_indices = []

    for idx, (is_a, is_b) in enumerate(zip(correct_a, correct_b)):
        # Restrict matching to the first max_idx samples
        if idx >= max_idx:
            break

        if mode == "a_only" and (is_a and not is_b):
            matching_indices.append(idx)
        elif mode == "b_only" and (is_b and not is_a):
            matching_indices.append(idx)
        elif mode == "all_correct" and (is_a and is_b):
            matching_indices.append(idx)
        elif mode == "all_wrong" and (not is_a and not is_b):
            matching_indices.append(idx)

    if not matching_indices:
        print(f"Warning: No samples found matching mode '{mode}' within first {max_idx} images.")
        return []

    if seed is not None:
        random.seed(seed)

    num_samples = min(num_samples, len(matching_indices))
    return random.sample(matching_indices, k=num_samples)

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


def main():
    dataset = "imagenet"
    split = "test_r"
    data_path = "./data"
    seed = 71397589
    input_size = 224
    cache_base_dir = "/home/stud/nemmler/retristyle/data/augmented_cache"

    dino_mapping_path = "/home/stud/nemmler/retristyle/results/retrieval_mapping/imagenet/retrieval_mapping_dino_test_r_s71397589.json"
    random_mapping_path = "/home/stud/nemmler/retristyle/results/retrieval_mapping/imagenet/retrieval_mapping_random_test_r_s71397589.json"

    dino_folder = "adain_dino_imagenet_test_r_s71397589"
    random_folder = "random_imagenet_test_r_s71397589"

    view_name = "view_001.png"
    output_path = "retrieval_comparison_roster.png"

    pred_path_dino = "/home/stud/nemmler/retristyle/results/ablation/adain_tta/tta_inference/predictions/imagenet/test_r/densenet121_adain_tta_zero_dino_nrefs32_seed71397589.json"
    pred_path_random = "/home/stud/nemmler/retristyle/results/ablation/adain_tta/tta_inference/predictions/imagenet/test_r/densenet121_adain_tta_zero_random_nrefs32_seed71397589.json"
    comparison_mode = "all_correct"

    num_examples = 3
    num_cols = num_examples * 2
    num_rows = 3

    raw_test_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split=split,
        transform=None,
    )

    test_set = create_dataset(
        dataset_name=dataset,
        data_path=data_path,
        split=split,
        transform=roster_transform,
    )

    ref_db = build_reference_db(
        dataset=dataset,
        data_path=data_path,
        input_size=input_size,
        seed=seed,
        split=split,
        show=True,
    )

    with open(dino_mapping_path, "r") as f:
        dino_mapping = json.load(f)

    with open(random_mapping_path, "r") as f:
        random_mapping = json.load(f)

    sample_indices = get_comparison_indices(
        path_a=pred_path_dino,
        path_b=pred_path_random,
        num_samples=num_examples,
        mode=comparison_mode,
        seed=67,
    )

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 2.0, num_rows * 2.0)
    )

    row_titles = ["Style", "Content", "AdaIN Output"]

    for i, sample_idx in enumerate(sample_indices):
        sample_str_idx = str(sample_idx)
        col_dino = i * 2
        col_random = i * 2 + 1

        # Content image load & prep
        raw_content_item = raw_test_set[sample_idx]
        raw_content_img = (
            raw_content_item[0]
            if isinstance(raw_content_item, (tuple, list))
            else raw_content_item
        )
        if not isinstance(raw_content_img, Image.Image):
            raw_content_img = tensor_to_pil(raw_content_img)

        content_item = test_set[sample_idx]
        content_img = (
            content_item[0]
            if isinstance(content_item, (tuple, list))
            else content_item
        )
        content_img = load_and_standardize(content_img, target_size=(224, 224))

        # --- Row 0: Style Images ---
        dino_ref_idx = dino_mapping[sample_str_idx][0][0]
        dino_style_img = load_and_standardize(
            ref_db.get_image(dino_ref_idx), target_size=(224, 224)
        )
        axes[0, col_dino].imshow(dino_style_img)
        axes[0, col_dino].axis("off")

        rand_ref_idx = random_mapping[sample_str_idx][0][0]
        rand_style_img = load_and_standardize(
            ref_db.get_image(rand_ref_idx), target_size=(224, 224)
        )
        axes[0, col_random].imshow(rand_style_img)
        axes[0, col_random].axis("off")

        # --- Row 1: Shared Content Image (Spans col_dino and col_random) ---
        axes[1, col_dino].axis("off")
        axes[1, col_random].axis("off")

        gs = axes[1, col_dino].get_gridspec()
        ax_content_merged = fig.add_subplot(gs[1, col_dino : col_random + 1])
        ax_content_merged.imshow(content_img)
        ax_content_merged.axis("off")

        # --- Row 2: Stylized Outputs ---
        folder_idx = f"{sample_idx:05d}"

        # DINO Stylized Output
        dino_path = Path(cache_base_dir) / dino_folder / folder_idx / view_name
        raw_dino = load_image_safe(dino_path)
        crop_dino = crop_to_content_aspect_ratio(raw_dino, raw_content_img)
        final_dino = load_and_standardize(crop_dino, target_size=(224, 224))
        axes[2, col_dino].imshow(final_dino)
        axes[2, col_dino].axis("off")

        # Random Stylized Output
        rand_path = Path(cache_base_dir) / random_folder / folder_idx / view_name
        raw_rand = load_image_safe(rand_path)
        crop_rand = crop_to_content_aspect_ratio(raw_rand, raw_content_img)
        final_rand = load_and_standardize(crop_rand, target_size=(224, 224))
        axes[2, col_random].imshow(final_rand)
        axes[2, col_random].axis("off")

    # Labels placed on the left side
    for row in range(num_rows):
        axes[row, -1].text(
            1.05,
            0.5,
            row_titles[row],
            transform=axes[row, -1].transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

    plt.subplots_adjust(wspace=0.04, hspace=0.04)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=300)


if __name__ == "__main__":
    main()
