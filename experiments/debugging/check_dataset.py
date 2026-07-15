from experiments.data import create_dataset
from torchvision.transforms import v2
import torch

# Assuming you added Camelyon17WILDS to your create_dataset factory
def debug_camelyon_structure():
    dataset_name = "camelyon17wilds"
    root_dir = "./data"
    
    # Check all three splits
    splits = ["train", "val", "test"]
    
    for split in splits:
        print(f"\n--- Checking Split: {split} ---")
        ds = create_dataset(dataset_name, root_dir, split)
        print(f"Dataset length: {len(ds)}")
        
        # Check first sample
        img, target = ds[0]
        print(f"Sample type: {type(img)}, Target type: {type(target)}")
        
        # Check hospital distribution for this split
        if hasattr(ds.dataset, 'metadata_array'):
            # metadata_array columns are ['hospital', 'slide', 'y']
            hospitals = ds.dataset.metadata_array[:, 0]
            unique_hospitals = torch.unique(hospitals)
            print(f"Hospitals present in this split: {unique_hospitals.tolist()}")
if __name__ == "__main__":
    debug_camelyon_structure()
