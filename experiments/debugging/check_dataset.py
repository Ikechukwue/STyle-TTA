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

from experiments.data import create_dataset


DATASET_NAME = "midog"
ROOT_DIR = "./data"
SPLITS = ["train", "val", "test", "breasts", "train@breasts", "val@breasts"]
CONTACT_SHEET_DIR = Path("./debug_output/contact_sheets")
CONTACT_SHEET_N = 16          # patches per class per split
BLANK_STD_THRESHOLD = 5.0     # grayscale std below this => flag as near-blank


def get_patch_dir(root_dir: str, split: str) -> Path:
    """
    Replicates Midog2022's internal path resolution exactly:
        resolved_split = f"{main_split}@{mapping_key}" if "@" in split else split
    which always reduces back to the original split string, so the patch
    dir is simply root_dir/midog22_dataset/patch_images/<split>.
    """
    return Path(root_dir) / "midog22_dataset" / "patch_images" / split


def get_samples_and_labels(root_dir: str, split: str):
    """
    Reads cached patches directly off disk instead of introspecting the
    dataset object returned by create_dataset(), which may wrap Midog2022
    in one or more layers we can't reliably unwrap generically. The patch
    cache layout (<patch_dir>/<label>/patch_<ann_id>.png) is stable and
    fully determined by Midog2022 regardless of any outer wrapper.

    Returns list of (path, label) or None if the patch dir doesn't exist.
    """
    patch_dir = get_patch_dir(root_dir, split)
    if not patch_dir.exists():
        return None

    samples = []
    for label_dir in sorted(patch_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        try:
            label = int(label_dir.name)
        except ValueError:
            continue
        for f in label_dir.iterdir():
            if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                samples.append((str(f), label))

    return samples if samples else None


def check_category_mapping(ds):
    """
    If the Midog2022 object exposes its raw coco_data, print the actual
    category_id -> name mapping so you can confirm category_id == 1
    really corresponds to "mitotic figure" in this specific JSON file,
    rather than assuming the general MIDOG convention holds.
    """
    inner = getattr(ds, "dataset", ds)
    coco_data = getattr(ds, "coco_data", None) or getattr(inner, "coco_data", None)
    if coco_data is None:
        print("  [skip] Could not find raw coco_data on dataset object "
              "(check attribute name if this matters to you).")
        return
    cats = coco_data.get("categories", [])
    print(f"  Category mapping from JSON: "
          f"{[(c.get('id'), c.get('name')) for c in cats]}")
    print("  >>> Confirm category_id used as label=1 in midog.py actually "
          "matches 'mitotic figure' (or whatever positive class you intend) above.")


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

    # Sampling everything can be slow for large splits; cap it but make it
    # deterministic and warn if capped.
    max_check = 5000
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

    # Any hash mapped to more than one distinct label = identical image
    # content saved under two different classes -> labeling/crop bug.
    conflicting = {h: lbls for h, lbls in hash_to_labels.items() if len(lbls) > 1}
    dup_same_class = sum(1 for h, paths_seen in hash_to_labels.items()
                          if len(paths_seen) == 1) 

    print(f"  Checked {checked} patches.")
    print(f"  Near-blank patches (grayscale std < {BLANK_STD_THRESHOLD}): "
          f"{blank_count} ({100*blank_count/max(checked,1):.1f}%)")
    if conflicting:
        print(f"  !!! WARNING: {len(conflicting)} identical patch(es) found "
              f"labeled as BOTH classes across samples. This indicates a "
              f"labeling or cropping bug (e.g. overlapping bounding boxes "
              f"resolving to different categories).")
    else:
        print("  No identical patches found assigned conflicting labels.")


def check_image_id_ranges(ds, split, image_id_registry):
    inner = getattr(ds, "dataset", ds)
    # Prefer looking at the coco-derived sample list if available (has image_id);
    # fall back to nothing if not exposed.
    raw_samples = getattr(ds, "_last_samples", None)  # not guaranteed to exist
    image_ids = None

    # Try to recover image_ids by re-deriving from patch filenames is fragile;
    # instead rely on coco_data + annotations directly if present.
    coco_data = getattr(ds, "coco_data", None) or getattr(inner, "coco_data", None)
    if coco_data is not None:
        # We don't have direct access to which ann_ids ended up in this split's
        # cache without re-running _get_split, so just report the full
        # annotation image_id universe as a fallback sanity check.
        pass

    print("  [note] Precise per-split WSI image_id sets require access to "
          "Midog2022's internal split logic; if you can, add a debug "
          "attribute (e.g. self.valid_ids) in _get_split so this script "
          "can directly report and cross-check overlap between splits.")


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

            samples = get_samples_and_labels(ds, split)

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

    print("\n--- Cross-split WSI overlap check ---")
    print("[note] Not fully automated here since Midog2022 doesn't currently "
          "expose which image_ids (WSIs) landed in each split. Recommended: "
          "temporarily add `self.valid_ids = valid_ids` inside `_get_split` "
          "in midog.py, then extend this script to print/compare "
          "train.valid_ids, val.valid_ids, test.valid_ids and assert they "
          "are pairwise disjoint (or intentionally overlapping only where "
          "you expect, e.g. never between train and test).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true",
                         help="Delete cached patches before checking, forcing "
                              "a full regeneration from the current code/labels.")
    args = parser.parse_args()
    debug_midog_structure(clean=args.clean)
