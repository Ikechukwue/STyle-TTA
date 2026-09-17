import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import v2
from code.config.helpers import get_top_k, get_names, load_json, get_class_name
from code.experiments.data import create_dataset
from code.experiments.tta.reference_db_setup import build_reference_db
import random
import numpy as np

list_of_ok_ones = [
    29680,
    #26467,
    #12904,
    3181,
    26889,
    25190
]
def get_comparison_indices(
    path_a: str | Path,
    path_b: str | Path,
    path_c: str | Path,
    num_samples: int = 4,
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

    # Get top-k predictions
    indices_a, _ = get_top_k(data_a, k=k)
    indices_b, _ = get_top_k(data_b, k=k)
    indices_c, _ = get_top_k(data_c, k=k)

    # Extract ground truth targets
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

    valid_modes = {
        # Strict single-model correctness (1 correct, 2 wrong)
        "a_only",
        "b_only",
        "c_only",
        # Strict pair-model correctness (2 correct, 1 wrong)
        "a_and_b_only",
        "a_and_c_only",
        "b_and_c_only",
        # All or none (3 correct or 3 wrong)
        "all_correct",
        "all_wrong",
        # Loose conditions (ignore third model status)
        "a_correct_b_wrong",
        "b_correct_a_wrong",
        "c_correct_a_wrong",
        "c_correct_b_wrong",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown mode: {mode}. Expected one of: {sorted(list(valid_modes))}"
        )

    for idx, (is_a, is_b, is_c) in enumerate(
        zip(correct_a, correct_b, correct_c)
    ):
        # --- Strict combinations ---
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

        # --- Loose pairwise combinations ---
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


def draw_info_card(ax, lines):
    """Turn an axis into a plain text info card."""
    ax.axis("off")
    text = "\n".join(lines)
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
        wrap=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create Image Roster with Style and Content on the Left"
    )
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--split", type=str, default="test_r")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=71397589)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument(
        "--mapping_path",
        type=str,
        default="/home/stud/nemmler/retristyle/results/retrieval_mapping/imagenet/retrieval_mapping_dino_test_r_s71397589.json",
    )
    parser.add_argument(
        "--cache_base_dir",
        type=str,
        default="/home/stud/nemmler/retristyle/data/augmented_cache",
    )
    parser.add_argument(
        "--stylized_folders",
        nargs="+",
        default=[
            "adain_dino_imagenet_test_r_s71397589",
            "dino_imagenet_test_r_s71397589",
        ],
    )
    parser.add_argument("--view_name", type=str, default="view_001.png")
    parser.add_argument("--output_path", type=str, default="image_roster1.png")

    # Prediction files used both for picking comparison samples and (optionally)
    # for reading off the predicted label of each stylized column.
    parser.add_argument(
        "--pred_path_a",
        type=str,
        default="/home/stud/nemmler/retristyle/results/ablation/retristyle/tta_inference/predictions/imagenet/test_r/densenet121_retristyle_vanilla_dino_nrefs2_seed71397589.json",
        help="Predictions file for model A, used for the comparison-mode sample "
        "selection and, if --info_cards is set, as the source of the predicted "
        "label shown for the corresponding entry in --stylized_folders.",
    )
    parser.add_argument(
        "--pred_path_b",
        type=str,
        default="/home/stud/nemmler/retristyle/results/ablation/adain_tta/tta_inference/predictions/imagenet/test_r/densenet121_adain_tta_zero_dino_nrefs2_seed71397589.json",
        help="Predictions file for model B, used the same way as --pred_path_a.",
    )
    parser.add_argument(
        "--pred_path_c",
        type=str,
        default="/home/stud/nemmler/retristyle/results/geometric_tta/tta_inference/predictions/imagenet/test_r/densenet121_geometric_vanilla_nviews1_seed71397589.json",
    )
    parser.add_argument(
        "--comparison_mode",
        type=str,
        default="b_correct_a_wrong",
    )

    # New: optional info cards next to each image (doubles the number of columns).
    parser.add_argument(
        "--info_cards",
        action="store_true",
        help="If set, add a text info-card column next to every image column "
        "(4x4 -> 4x8). Style/content columns show 'True label: X', stylized "
        "columns show 'Pred label: Y' for the corresponding prediction file.",
    )

    args = parser.parse_args()

    # --prediction_paths[i] must correspond to --stylized_folders[i]. We default
    # to [pred_path_a, pred_path_b], matching the original hardcoded behaviour.
    prediction_paths = [args.pred_path_b, args.pred_path_a]

    if args.info_cards and len(prediction_paths) != len(args.stylized_folders):
        raise ValueError(
            "When --info_cards is set, the number of prediction files "
            f"({len(prediction_paths)}: pred_path_a, pred_path_b) must match "
            f"the number of --stylized_folders ({len(args.stylized_folders)}). "
            "Pass a matching --pred_path_a/--pred_path_b or adjust --stylized_folders."
        )

    num_rows = 4
    num_stylized_cols = len(args.stylized_folders)
    num_base_cols = 2 + num_stylized_cols
    num_cols = num_base_cols * 2 if args.info_cards else num_base_cols

    eval_split = args.split

    raw_test_set = create_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split=eval_split,
        transform=None,
    )

    test_set = create_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split=eval_split,
        transform=roster_transform,
    )

    ref_db = build_reference_db(
        dataset=args.dataset,
        data_path=args.data_path,
        input_size=args.input_size,
        seed=args.seed,
        split=args.split,
        show=True,
    )

    with open(args.mapping_path, "r") as f:
        retrieval_mapping = json.load(f)

    # Compare two predictions and retrieve `num_rows` indices matching the
    # requested comparison mode (e.g. Model A succeeded but Model B failed).
    #sample_indices = list_of_ok_ones
    
    sample_indices = get_comparison_indices(
        path_a=args.pred_path_a,
        path_b=args.pred_path_b,
        path_c=args.pred_path_c,
        num_samples=num_rows,
        mode=args.comparison_mode,
        seed=3,
    )
    
    # If info cards are requested, pre-load true/predicted labels for lookup.
    y_true_all = None
    pred_top1_by_folder = None

    pred_data_by_folder = []
    for p in [args.pred_path_b, args.pred_path_a ,args.pred_path_c]:
        with open(p, "r") as f:
            pred_data_by_folder.append(json.load(f))

    # y_true is identical across prediction files (already checked in
    # get_comparison_indices), so any one of them works here.
    y_true_all = np.array(
        [s["y_true"] for s in pred_data_by_folder[0]["predictions"]]
    )

    pred_top1_by_folder = []
    for pred_data in pred_data_by_folder:
        indices_top1, _ = get_top_k(pred_data, k=1)
        pred_top1_by_folder.append(np.asarray(indices_top1).reshape(-1))

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3)
    )

    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)

    col_labels = ["Style", "Content"]  

    for folder in args.stylized_folders:
        fname = folder.split("_")[0].upper()
        if fname == "ADAIN":
            m_name = "AdaIN"
        if fname == "DINO":
            m_name = "StyleID"

        col_labels.append(m_name)
 
    def img_ax(row, base_col):
        return axes[row, 2 * base_col] if args.info_cards else axes[row, base_col]

    def info_ax(row, base_col):
        return axes[row, 2 * base_col + 1] if args.info_cards else None
    class_list = get_names()

    for row, sample_idx in enumerate(sample_indices):
        sample_str_idx = str(sample_idx)

        raw_content_item = raw_test_set[sample_idx]
        raw_content_img = (
            raw_content_item[0]
            if isinstance(raw_content_item, (tuple, list))
            else raw_content_item
        )
        if not isinstance(raw_content_img, Image.Image):
            raw_content_img = tensor_to_pil(raw_content_img)

        true_label = int(y_true_all[sample_idx]) 
        true_name = get_class_name(true_label, class_list)

        # ---- Column 0: Style Image ----
        style_ref_idx = retrieval_mapping[sample_str_idx][0][0]
        style_ref_class = retrieval_mapping[sample_str_idx][0][1]
        style_img_tensor = ref_db.get_image(style_ref_idx)
        style_img = load_and_standardize(
            style_img_tensor, target_size=(224, 224)
        )
        style_name = get_class_name(style_ref_class, class_list)
        ax = img_ax(row, 0)
        ax.imshow(style_img)
        ax.axis("off")
        if args.info_cards:
            draw_info_card(info_ax(row, 0), [f"True label: {style_name}"])
        else:
            ax.text(
                0.5, -0.04,
                f"{style_name}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
        # ---- Column 1: Content Image ----
        content_item = test_set[sample_idx]
        content_img = (
            content_item[0]
            if isinstance(content_item, (tuple, list))
            else content_item
        )
        content_img = load_and_standardize(content_img, target_size=(224, 224))
        # Evaluate Content prediction (Path C) correctness
        content_pred_label = int(pred_top1_by_folder[2][sample_idx])  # Index 2 corresponds to Path C
        content_pred_name = get_class_name(content_pred_label, class_list)
        is_content_correct = content_pred_label == true_label

        content_color = "green" if is_content_correct else "red"
        content_status = "✓ Correct" if is_content_correct else "✗ False"
        content_text = f"True: {true_name}\n({content_status})"

        ax = img_ax(row, 1)
        ax.imshow(content_img)
        ax.axis("off")
        if args.info_cards:
            card_ax = info_ax(row, 1)
            card_ax.axis("off")
            card_ax.text(
                0.5, 0.5, f"True: {true_name}\nPred: {content_pred_name}",
                ha="center", va="center", fontsize=9, transform=card_ax.transAxes, wrap=True
            )
            card_ax.text(
                0.5, 0.2, content_status,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=content_color, transform=card_ax.transAxes
            )
        else:
            ax.text(
                0.5, -0.04, f"{true_name} ",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
            )
            ax.text(
                0.5, -0.04, f"[{content_status}]",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                color=content_color, fontweight="bold",
            )
        # ---- Columns 2 to 2+n: Stylized Outputs ----
        folder_idx = f"{sample_idx:05d}"
        for n_idx, folder_name in enumerate(args.stylized_folders):
            base_c = 2 + n_idx
            stylized_path = (
                Path(args.cache_base_dir)
                / folder_name
                / folder_idx
                / args.view_name
            )

            raw_stylized = load_image_safe(stylized_path)
            cropped_stylized = crop_to_content_aspect_ratio(
                raw_stylized, raw_content_img
            )
            final_stylized = load_and_standardize(
                cropped_stylized, target_size=(224, 224)
            )

            ax = img_ax(row, base_c)
            ax.imshow(final_stylized)
            ax.axis("off")
            pred_label = int(pred_top1_by_folder[n_idx][sample_idx])
            pred_name = get_class_name(pred_label, class_list)
            # Check model prediction against Ground Truth
            is_correct = pred_label == true_label
            status_text = "✓ Correct" if is_correct else "✗ False"
            text_color = "green" if is_correct else "red"

            if args.info_cards:
                card_ax = info_ax(row, base_c)
                card_ax.axis("off")
                card_ax.text(
                    0.5, 0.6, f"Pred: {pred_name}",
                    ha="center", va="center", fontsize=9, transform=card_ax.transAxes, wrap=True
                )
                card_ax.text(
                    0.5, 0.3, status_text,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=text_color, transform=card_ax.transAxes
                )
            else:
                ax.text(
                    0.5, -0.04, f"{pred_name} ",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                )
                ax.text(
                    0.5, -0.04, f"[{status_text}]",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    color=text_color, fontweight="bold",
                )
    for base_c in range(num_base_cols):
        ax = img_ax(num_rows - 1, base_c)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xlabel(
            col_labels[base_c], fontsize=12, fontweight="bold", labelpad=28
        )

    plt.tight_layout()
    plt.savefig(args.output_path, bbox_inches="tight", pad_inches=0.2, dpi=300)


if __name__ == "__main__":
    modes = [
        "a_only",
        "b_only",
        "c_only",
        "a_and_b_only",
        "a_and_c_only",
        "b_and_c_only",
        "all_correct",
        "all_wrong",
        "a_correct_b_wrong",
        "b_correct_a_wrong",
        "c_correct_a_wrong",
        "c_correct_b_wrong",
    ]
    ds = "imagenet"
    base_args = [
        "--dataset", f"{ds}",
        "--split", "test_r",
    ]

    for mode in modes:
        out_name = f"output/images/{ds}/roster_{mode}.png"
        print(f"Running mode: {mode} -> Output: {out_name}")

        import sys
        sys.argv = [
            "script.py",
            *base_args,
            "--comparison_mode", mode,
            "--output_path", out_name,
        ]


        main()

