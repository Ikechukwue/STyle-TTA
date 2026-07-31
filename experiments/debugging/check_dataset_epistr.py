"""
Sanity-check script for the Epithelium-Stroma dataset pipeline.

Verifies:
  1. Dataset length and sample load state
  2. Category class_to_idx mapping from underlying ImageFolder instances
  3. Class distribution per split (train=NKI, val=VGH, test=IHC)
  4. Duplicate patches assigned to different classes (leakage/labeling bug)
  5. Near-blank / degenerate patches (bad crops/corrupted data)
  6. A saved visual contact sheet per split/class for visual inspection

Run with:
    python -m experiments.debug_epithelium_stroma_structure
"""

import os
import sys
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder

from experiments.data import create_dataset


DATASET_NAME = "epistr"
ROOT_DIR = "./data"
SPLITS = ["train", "val", "test"]
CONTACT_SHEET_DIR = Path("./debug_output/contact_sheets_epithelium_stroma")
CONTACT_SHEET_N = 16          # patches per class per split
BLANK_STD_THRESHOLD = 5.0     # grayscale std below this => flag as near-blank


def get_samples_and_labels(ds: Dataset):
    """
    Retrieves (path, label) tuples by recursively unwrapping EpitheliumStroma,
    Subset, and ConcatDataset wrappers down to base ImageFolder instances.
    """
    if hasattr(ds, 'samples'):
        return ds.samples

    if isinstance(ds, Subset):
        base_samples = get_samples_and_labels(ds.dataset)
        return [base_samples[i] for i in ds.indices] if base_samples else None

    if isinstance(ds, ConcatDataset):
        combined_samples = []
        for sub_ds in ds.datasets:
            sub_samples = get_samples_and_labels(sub_ds)
            if sub_samples:
                combined_samples.extend(sub_samples)
        return combined_samples if combined_samples else None

    if hasattr(ds, 'dataset'):
        return get_samples_and_labels(ds.dataset)

    if isinstance(ds, ImageFolder):
        return ds.samples

    return None


def check_category_mapping(ds: Dataset):
    """
    Checks the category mapping by introspecting underlying ImageFolder objects.
    """
    base_ds = ds.dataset if hasattr(ds, 'dataset') else ds

    while isinstance(base_ds, Subset):
        base_ds = base_ds.dataset

    if isinstance(base_ds, ConcatDataset) and len(base_ds.datasets) > 0:
        base_ds = base_ds.datasets[0]

    while hasattr(base_ds, 'dataset') and not isinstance(base_ds, ImageFolder):
        base_ds = base_ds.dataset

    if hasattr(base_ds, 'class_to_idx'):
        print(f"  Class mapping (folder -> index): {base_ds.class_to_idx}")
    else:
        print("  [skip] Could not inspect class_to_idx mapping.")


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


def debug_epithelium_stroma_structure():
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
    debug_epithelium_stroma_structure()
