import json
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image

from config.helpers import get_class_name, get_names, get_top_k
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


def get_comparison_indices(
    path_a: str | Path,
    path_b: str | Path,
    path_c: str | Path,
    num_samples: int = 8,
    mode: str = "c_only",
    seed: int | None = None,
    k: int = 1,
) -> list[int]:
    with open(path_a, "r") as f:
        data_a = json.load(f)
    with open(path_b, "r") as f:
        data_b = json.load(f)
    with open(path_c, "r") as f:
        data_c = json.load(f)

    indices_a, _ = get_top_k(data_a, k=k)
    indices_b, _ = get_top_k(data_b, k=k)
    indices_c, _ = get_top_k(data_c, k=k)

    y_true_a = np.array([s["y_true"] for s in data_a["predictions"]])
    y_true_b = np.array([s["y_true"] for s in data_b["predictions"]])
    y_true_c = np.array([s["y_true"] for s in data_c["predictions"]])

    if not (
        np.array_equal(y_true_a, y_true_b)
        and np.array_equal(y_true_b, y_true_c)
    ):
        raise ValueError(
            "path_a, path_b, and path_c contain different y_true values."
        )

    correct_a = np.any(indices_a == y_true_a[:, None], axis=-1)
    correct_b = np.any(indices_b == y_true_b[:, None], axis=-1)
    correct_c = np.any(indices_c == y_true_c[:, None], axis=-1)

    matching_indices = []

    for idx, (is_a, is_b, is_c) in enumerate(
        zip(correct_a, correct_b, correct_c)
    ):
        if mode == "a_only" and (is_a and not is_b and not is_c):
            matching_indices.append(idx)
        elif mode == "b_only" and (is_b and not is_a and not is_c):
            matching_indices.append(idx)
        elif mode == "c_only" and (is_c and not is_a and not is_b):
            matching_indices.append(idx)
        elif mode == "a_and_b_only" and (is_a and is_b and not is_c):
            matching_indices.append(idx)
        elif mode == "a_and_c_only" and (is_a and is_c and not is_b):
            matching_indices.append(idx)
        elif mode == "b_and_c_only" and (is_b and is_c and not is_a):
            matching_indices.append(idx)
        elif mode == "all_correct" and (is_a and is_b and is_c):
            matching_indices.append(idx)
        elif mode == "all_wrong" and (not is_a and not is_b and not is_c):
            matching_indices.append(idx)
        elif mode == "a_correct_b_wrong" and (is_a and not is_b):
            matching_indices.append(idx)
        elif mode == "b_correct_a_wrong" and (is_b and not is_a):
            matching_indices.append(idx)
        elif mode == "c_correct_a_wrong" and (is_c and not is_a):
            matching_indices.append(idx)
        elif mode == "c_correct_b_wrong" and (is_c and not is_b):
            matching_indices.append(idx)

    if not matching_indices:
        print(f"Warning: No samples found matching mode '{mode}'.")
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
    mapping_path = "/home/stud/nemmler/retristyle/results/retrieval_mapping/imagenet/retrieval_mapping_dino_test_r_s71397589.json"
    cache_base_dir = "/home/stud/nemmler/retristyle/data/augmented_cache"
    stylized_folders = [
        "adain_dino_imagenet_test_r_s71397589",
        "dino_imagenet_test_r_s71397589",
    ]
    view_name = "view_001.png"
    output_path = "image_roster_clea.png"

    pred_path_a = "/home/stud/nemmler/retristyle/results/ablation/retristyle/tta_inference/predictions/imagenet/test_r/densenet121_retristyle_vanilla_dino_nrefs2_seed71397589.json"
    pred_path_b = "/home/stud/nemmler/retristyle/results/ablation/adain_tta/tta_inference/predictions/imagenet/test_r/densenet121_adain_tta_zero_dino_nrefs2_seed71397589.json"
    pred_path_c = "/home/stud/nemmler/retristyle/results/geometric_tta/tta_inference/predictions/imagenet/test_r/densenet121_geometric_vanilla_nviews1_seed71397589.json"
    comparison_mode = "a_and_b_only"

    num_cols = 8
    num_rows = 4

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

    with open(mapping_path, "r") as f:
        retrieval_mapping = json.load(f)

    sample_indices = get_comparison_indices(
        path_a=pred_path_a,
        path_b=pred_path_b,
        path_c=pred_path_c,
        num_samples=num_cols,
        mode=comparison_mode,
        seed=67777,
    )

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 2.0, num_rows * 2.0)
    )

    row_titles = ["Style", "Content", "AdaIN", "StyleID"]

    for col, sample_idx in enumerate(sample_indices):
        sample_str_idx = str(sample_idx)

        raw_content_item = raw_test_set[sample_idx]
        raw_content_img = (
            raw_content_item[0]
            if isinstance(raw_content_item, (tuple, list))
            else raw_content_item
        )
        if not isinstance(raw_content_img, Image.Image):
            raw_content_img = tensor_to_pil(raw_content_img)

        # --- Row 0: Style Image ---
        style_ref_idx = retrieval_mapping[sample_str_idx][0][0]
        style_img_tensor = ref_db.get_image(style_ref_idx)
        style_img = load_and_standardize(
            style_img_tensor, target_size=(224, 224)
        )

        ax_style = axes[0, col]
        ax_style.imshow(style_img)
        ax_style.axis("off")

        # --- Row 1: Content Image ---
        content_item = test_set[sample_idx]
        content_img = (
            content_item[0]
            if isinstance(content_item, (tuple, list))
            else content_item
        )
        content_img = load_and_standardize(content_img, target_size=(224, 224))

        ax_content = axes[1, col]
        ax_content.imshow(content_img)
        ax_content.axis("off")

        # --- Row 2 & Row 3: Stylized Outputs ---
        folder_idx = f"{sample_idx:05d}"
        for n_idx, folder_name in enumerate(stylized_folders):
            row_target = 2 + n_idx
            stylized_path = (
                Path(cache_base_dir)
                / folder_name
                / folder_idx
                / view_name
            )

            raw_stylized = load_image_safe(stylized_path)
            cropped_stylized = crop_to_content_aspect_ratio(
                raw_stylized, raw_content_img
            )
            final_stylized = load_and_standardize(
                cropped_stylized, target_size=(224, 224)
            )

            ax_stylized = axes[row_target, col]
            ax_stylized.imshow(final_stylized)
            ax_stylized.axis("off")

    # Labels placed explicitly on the right side of each row
    for row in range(num_rows):
        axes[row, 0].text(
            -0.05,
            0.5,
            row_titles[row],
            transform=axes[row, 0].transAxes,
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="center",
        )

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=300)


if __name__ == "__main__":
    main()
