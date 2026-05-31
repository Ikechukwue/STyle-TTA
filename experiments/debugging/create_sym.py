import os
from pathlib import Path

def create_imagenet_r_subset_symlinks():
    # Define source and destination paths
    src_dir = Path("/home/stud/nemmler/retristyle/data/imagenet/imagenetr/imagenet-r")
    dst_root = Path("/home/stud/nemmler/retristyle/data/imagenet/imagenetr/imagenet-c26")

    print(f"Scanning source directory: {src_dir}")
    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist.")
        return

    # Gather and sort all class directories (e.g., n01443537, n01614990...)
    # Excludes files to ensure we only grab folders
    all_classes = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    
    # Slice the first 26 classes
    target_classes = all_classes[:26]
    
    print(f"Found {len(all_classes)} total classes. Processing the first {len(target_classes)} classes...")

    # Ensure the destination parent folder exists
    dst_root.mkdir(parents=True, exist_ok=True)

    links_created = 0

    for class_path in target_classes:
        class_name = class_path.name
        new_class_dir = dst_root / class_name
        
        # Create the specific class folder inside your new destination root
        new_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Iterate over every image inside the original class folder
        for img_path in class_path.iterdir():
            if img_path.is_file():
                link_path = new_class_dir / img_path.name
                
                # If a symlink already exists, remove it to prevent collision errors on rerun
                if link_path.is_symlink() or link_path.exists():
                    link_path.unlink()
                
                # Create a relative symlink from the new path pointing back to the original file
                # Using os.path.relpath keeps the symlinks flexible and robust
                relative_target = os.path.relpath(img_path, start=new_class_dir)
                link_path.symlink_to(relative_target)
                links_created += 1

    print(f"\nSuccessfully finished!")
    print(f"Created {len(target_classes)} class folders at: {dst_root}")
    print(f"Total symlinks generated: {links_created}")

if __name__ == "__main__":
    create_imagenet_r_subset_symlinks()
