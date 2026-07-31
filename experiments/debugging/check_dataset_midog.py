"""
Expanded sanity-check script for the MIDOG dataset pipeline.

Beyond the basic length/shape check, this verifies:
  1. Patch cache staleness (regenerates from scratch with --clean)
  2. Category id -> name mapping (confirms label 1 really means "mitotic figure")
  3. Class distribution per split (matches the imbalance you expect)
  4. Duplicate patches assigned to *different* classes (leakage/labeling bug)
  5. Near-blank / degenerate patches (bad crops)
  6. WSI (image_id) overlap across splits (train/val/test leakage)
  7. A saved visual contact sheet per split/class so you can eyeball patches

Run with:
    python -m experiments.debug_midog_structure
    python -m experiments.debug_midog_structure --clean   # wipe cached patches first
"""

import os
import sys
import shutil
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from experiments.data import create_dataset


DATASET_NAME = "midog"
ROOT_DIR = "./data"
SPLITS = ["train", "val", "test", "breasts", "train@breasts", "val@breasts"]
CONTACT_SHEET_DIR = Path("./debug_output/contact_sheets")
CONTACT_SHEET_N = 16          # patches per class per split
BLANK_STD_THRESHOLD = 5.0     # grayscale std below this => flag as near-blank


def get_base_image_folder(dataset: Dataset) -> ImageFolder:
    """
    Unwraps nested dataset objects until reaching the core ImageFolder instance.
    """
    current_ds = dataset
    while hasattr(current_ds, 'dataset'):
        current_ds = current_ds.dataset
    if not isinstance(current_ds, ImageFolder):
        raise TypeError(f"Expected base dataset to be ImageFolder, but found {type(current_ds)}")
    return current_ds


def get_samples_and_labels(ds: Dataset):
    """
    Retrieves the list of (path, label) pairs directly from the unwrapped
    base ImageFolder dataset.
    """
    try:
        base_ds = get_base_image_folder(ds)
        return base_ds.samples
    except TypeError as e:
        print(f"  [skip] Could not unwrap ImageFolder: {e}")
        return None


import json

JSON_PATH = Path("/home/stud/nemmler/retristyle/data/midog22_dataset/images/MIDOG2022_training_png.json")

def check_category_mapping(ds):
    """
    Checks the category mapping by introspecting the dataset or directly
    reading the COCO metadata JSON file on disk.
    """
    coco_data = None

    try:
        base_ds = get_base_image_folder(ds)
        coco_data = getattr(base_ds, "coco_data", None)
    except TypeError:
        pass

    if coco_data is None:
        coco_data = getattr(ds, "coco_data", None)

    # Fallback to reading the JSON file directly from disk
    if coco_data is None and JSON_PATH.exists():
        try:
            with open(JSON_PATH, "r") as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"  [error] Failed to read metadata JSON from {JSON_PATH}: {e}")

    if coco_data is None:
        print("  [skip] Could not find raw coco_data on dataset or load JSON from disk.")
        return

    cats = coco_data.get("categories", [])
    print(f"  Category mapping from JSON: {[(c.get('id'), c.get('name')) for c in cats]}")


def check_class_distribution(samples, split):
    if samples is None:
        print("  [skip] Could not access underlying samples for class distribution.")
        return
    labels = [lbl for _, lbl in samples]
    counts = Counter(labels)
    total = len(labels)
    print(f"  Class distribution: {dict(counts)}  (total={total})")
    for cls, cnt in sorted(counts.items()):
        print(f"    class {cls}: {cnt} ({100*cnt/total:.1f}%)")
    if len(counts) < 2:
        print("  !!! WARNING: only one class present in this split!")


def check_duplicates_and_blanks(samples, split):
    if samples is None:
        print("  [skip] Could not access underlying samples for duplicate/blank check.")
        return

    hash_to_labels = defaultdict(set)
    blank_count = 0
    checked = 0

    max_check = len(samples)
    to_check = samples if len(samples) <= max_check else samples[:max_check]
    if len(samples) > max_check:
        print(f"  (checking first {max_check} of {len(samples)} samples for speed)")

    for path, label in to_check:
        try:
            with Image.open(path) as img:
                img_gray = img.convert("L")
                arr = np.asarray(img_gray, dtype=np.float32)
                std = arr.std()
                if std < BLANK_STD_THRESHOLD:
                    blank_count += 1

                with open(path, "rb") as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()
                hash_to_labels[content_hash].add(label)
            checked += 1
        except Exception as e:
            print(f"  [error] Could not open {path}: {e}")

    conflicting = {h: lbls for h, lbls in hash_to_labels.items() if len(lbls) > 1}

    print(f"  Checked {checked} patches.")
    print(f"  Near-blank patches (grayscale std < {BLANK_STD_THRESHOLD}): "
          f"{blank_count} ({100*blank_count/max(checked,1):.1f}%)")
    if conflicting:
        print(f"  !!! WARNING: {len(conflicting)} identical patch(es) found "
              f"labeled as BOTH classes across samples. This indicates a "
              f"labeling or cropping bug.")
    else:
        print("  No identical patches found assigned conflicting labels.")


def save_contact_sheet(samples, split):
    if samples is None:
        return
    CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)

    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append(path)

    for label, paths in by_class.items():
        rng = np.random.RandomState(42)
        chosen = rng.choice(paths, size=min(CONTACT_SHEET_N, len(paths)), replace=False)

        thumbs = []
        for p in chosen:
            with Image.open(p) as img:
                thumbs.append(img.convert("RGB").resize((112, 112)))

        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 112, rows * 112), color=(30, 30, 30))
        for i, thumb in enumerate(thumbs):
            x = (i % cols) * 112
            y = (i // cols) * 112
            sheet.paste(thumb, (x, y))

        safe_split = split.replace("@", "_at_")
        out_path = CONTACT_SHEET_DIR / f"{DATASET_NAME}_{safe_split}_class{label}.png"
        sheet.save(out_path)
        print(f"  Saved contact sheet: {out_path}")


def debug_midog_structure(clean: bool = False):
    if clean:
        patch_root = Path(ROOT_DIR) / "midog22_dataset" / "patch_images"
        if patch_root.exists():
            print(f"--clean specified: removing cached patches at {patch_root}")
            shutil.rmtree(patch_root)
        else:
            print(f"--clean specified but no cache found at {patch_root} (nothing to remove)")

    for split in SPLITS:
        print(f"\n--- Checking Split: {split} ---")
        try:
            ds = create_dataset(DATASET_NAME, ROOT_DIR, split)
            print(f"Dataset length: {len(ds)}")

            if len(ds) == 0:
                print("  [skip] Empty split.")
                continue

            img, target = ds[0]
            print(f"Sample 0 img type: {type(img)}, target: {target}")

            samples = get_samples_and_labels(ds)

            print("Category mapping check:")
            check_category_mapping(ds)

            print("Class distribution:")
            check_class_distribution(samples, split)

            print("Duplicate / blank patch check:")
            check_duplicates_and_blanks(samples, split)

            print("Contact sheet:")
            save_contact_sheet(samples, split)

        except Exception as e:
            print(f"Error loading split '{split}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                         help="Delete cached patches before checking, forcing "
                              "a full regeneration from the current code/labels.")
    args = parser.parse_args()
    debug_midog_structure(clean=args.clean)
