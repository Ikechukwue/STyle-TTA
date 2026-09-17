import os
import random
from pathlib import Path
import json
from code.config.paths import DATA_PATH


def sync_imagenet_subset(subset_r_path, full_imagenet_path, output_path, subset_name,json_file_path):
    """
    1. Reads classes from an existing ImageNet-R subset.
    2. Links full ImageNet data for those classes to a new location.
    3. Updates a JSON file with the class list.
    """
    r_path = Path(subset_r_path)
    full_path = Path(full_imagenet_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Get the class IDs from your existing subset folders
    selected_classes = [d.name for d in r_path.iterdir() if d.is_dir()]
    print(f"Found {len(selected_classes)} classes in the ImageNet subset.")

    total_links = 0
    synced_classes = 0

    # 2. Create the subset from the full ImageNet set
    for cls_id in selected_classes:
        src_cls_dir = full_path / cls_id
        
        if not src_cls_dir.exists():
            print(f"Warning: Class {cls_id} not found in full ImageNet path. Skipping.")
            continue

        dst_cls_dir = out_path / cls_id
        dst_cls_dir.mkdir(exist_ok=True)

        # Link every image in the full ImageNet class folder
        images = [f for f in src_cls_dir.iterdir() if f.is_file()]
        for img in images:
            link_path = dst_cls_dir / img.name
            try:
                link_path.symlink_to(img.resolve())
                total_links += 1
            except FileExistsError:
                pass
        
        synced_classes += 1

    print(f"Linked {total_links} images across {synced_classes} classes to: {output_path}")

    # 3. Update the JSON file
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Ensure subset exists in the json
        if subset_name not in data:
            data[subset_name] = []
        
        # Add new classes, avoiding duplicates, and sort them
        existing_classes = set(data[subset_name])
        new_classes = [c for c in selected_classes if c not in existing_classes]
        
        if new_classes:
            data[subset_name].extend(new_classes)
            data[subset_name].sort() # Keeping it tidy
            print(f"Added {len(new_classes)} new classes to {json_file_path}.")
        else:
            print("No new classes to add to JSON.")

        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        print(f"JSON file not found at {json_file_path}. Skipping JSON update.")


def create_flexible_subset(src_root, dst_root, num_classes=10, target_images_per_class=100):
    """
    Creates a subset with a variable number of classes, aiming for a target number
    of images per class.
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)
    dst_path.mkdir(parents=True, exist_ok=True)

    all_classes = [d for d in src_path.iterdir() if d.is_dir()]
    
    # Cap num_classes to available folders
    num_to_select = min(len(all_classes), num_classes)
    selected_classes = random.sample(all_classes, num_to_select)
    
    print(f"--- Configuration ---")
    print(f"Targeting {num_to_select} classes with ~{target_images_per_class} images each.\n")

    distribution = {}
    total_links = 0

    for cls_folder in selected_classes:
        dst_cls_folder = dst_path / cls_folder.name
        dst_cls_folder.mkdir(exist_ok=True)

        images = [f for f in cls_folder.iterdir() if f.is_file()]
        if not images:
            continue

        # Logic: Take the target number OR everything available if it's less
        n_to_sample = min(len(images), target_images_per_class)
        sampled_images = random.sample(images, n_to_sample)
        
        distribution[cls_folder.name] = len(sampled_images)

        for img in sampled_images:
            link_path = dst_cls_folder / img.name
            try:
                link_path.symlink_to(img.resolve())
                total_links += 1
            except FileExistsError:
                pass

    # --- Summary Table ---
    print(f"{'Class Name':<20} | {'Images Linked':<15} | {'Status'}")
    print("-" * 55)
    for cls_name, count in sorted(distribution.items()):
        status = "Full Target" if count == target_images_per_class else "All Available (Under Target)"
        print(f"{cls_name:<20} | {count:<15} | {status}")
    
    print("-" * 55)
    print(f"Total Images in Subset: {total_links}")

# --- Configuration ---
SOURCE_DIR = str(DATA_PATH / "imagenet" / "imagenetr" / "imagenet-r")
DESTINATION_DIR = str(DATA_PATH / "imagenet" / "imagenettest")

if __name__ == "__main__":
    # Change num_classes to whatever you need (e.g., 20, 50, 200)
    # Change target_images_per_class to your preferred balance point
    """
    create_flexible_subset(
        SOURCE_DIR, 
        DESTINATION_DIR, 
        num_classes=15, 
        target_images_per_class=100
    )
    """
    # --- Configuration ---
    # Path to the ImageNet-R subset you just created
    SUBSET_R_DIR = str(DATA_PATH / "imagenet" / "imagenetabl")

    # Path to your REAL full ImageNet (the one with 1000 classes)
    FULL_IMAGENET_DIR = str(DATA_PATH / "imagenet" / "imagenet1k" / "ILSVRC" / "Data" / "CLS-LOC" / "train")

    # Where you want the full ImageNet images for these specific classes to go
    OUTPUT_SUBSET_DIR = str(DATA_PATH / "imagenet" / "imagenet1k" / "subsets" / "test_abl")

    # Path to your JSON file
    JSON_PATH = str(DATA_PATH / "imagenet" / "imagenet_subsets.json")

    sync_imagenet_subset(
        SUBSET_R_DIR, 
        FULL_IMAGENET_DIR, 
        OUTPUT_SUBSET_DIR,
        "test_abl", 
        JSON_PATH
    )
