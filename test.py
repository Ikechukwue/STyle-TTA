import os
import random
from pathlib import Path

def create_balanced_subset(src_root, dst_root, images_per_class=1):
    """
    Creates a balanced subset of a dataset using symbolic links.
    
    :param src_root: Path to the original ImageNet-R directory.
    :param dst_root: Path where the symlink subset will be created.
    :param images_per_class: Number of images to pick from each folder.
    """
    src_path = Path(src_root)
    dst_path = Path(dst_root)

    # Ensure the destination exists
    dst_path.mkdir(parents=True, exist_ok=True)

    # Get all subdirectories (classes)
    classes = [d for d in src_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(classes)} classes. Target: {images_per_class} image(s) per class.")

    total_links = 0

    for cls_folder in classes:
        # Get all image files in the class folder
        images = [f for f in cls_folder.iterdir() if f.is_file()]
        
        if not images:
            print(f"Warning: No images found in {cls_folder.name}. Skipping.")
            continue

        # Create corresponding class folder in destination
        (dst_path / cls_folder.name).mkdir(exist_ok=True)

        # Randomly sample images
        n_to_sample = min(len(images), images_per_class)
        sampled_images = random.sample(images, n_to_sample)

        for img in sampled_images:
            link_path = dst_path / cls_folder.name / img.name
            
            # Create the symlink (absolute path ensures the link doesn't break)
            try:
                link_path.symlink_to(img.resolve())
                total_links += 1
            except FileExistsError:
                pass

    print(f"Done! Created {total_links} symlinks across {len(classes)} classes at: {dst_root}")

# --- Configuration ---
SOURCE_DIR = "/home/stud/nemmler/retristyle/data/imagenet/imagenetr/imagenet-r"
DESTINATION_DIR = "/home/stud/nemmler/retristyle/data/imagenet/imagenettest"

if __name__ == "__main__":
    create_balanced_subset(SOURCE_DIR, DESTINATION_DIR, images_per_class=1)
