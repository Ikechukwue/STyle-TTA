import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from code.experiments.data import create_dataset
# Assuming EuroSAT and UCMerced are imported or defined above
# from datasets import EuroSAT, UCMerced 

class DummyArgs:
    def __init__(self):
        self.input_size = 224
        self.data_path = "."  # Root dir where 'data/satelite' lives
        self.num_workers = 2

def ResizeWhileRetainAspectRatio(size):
    return v2.Resize(size)

def worker_seed(worker_id):
    pass

def test_pipeline():
    args = DummyArgs()
    g = torch.Generator()
    g.manual_seed(42)

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        ResizeWhileRetainAspectRatio(size=args.input_size),
    ])

    print("--- Testing EuroSAT Dataset (Mapped) ---")
    try:
        # Added @mapped suffix
        eurosat_set = create_dataset(
            dataset_name="eurosat",
            data_path="./data",
            split="train@mapped", 
            transform=test_transform,
        )
        eurosat_loader = DataLoader(
            eurosat_set, batch_size=1, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=worker_seed, generator=g,
        )
        
        print(f"Total EuroSAT mapped samples: {len(eurosat_set)}")
        img, target = next(iter(eurosat_loader))
        print(f"EuroSAT Batch - Image shape: {img.shape}, Target: {target.item()}")
        print("EuroSAT test passed successfully.\n")
    except Exception as e:
        print(f"EuroSAT test failed: {e}\n")

    print("--- Testing UC Merced Dataset (Mapped) ---")
    try:
        # Added subset_name='mapped'
        ucmerced_set = create_dataset(
            dataset_name="ucmerced",
            data_path="./data",
            split="mapped", 
            transform=test_transform,
        )
        ucmerced_loader = DataLoader(
            ucmerced_set, batch_size=1, shuffle=False,
            num_workers=args.num_workers, worker_init_fn=worker_seed, generator=g,
        )
        
        print(f"Total UC Merced mapped samples: {len(ucmerced_set)}")
        img, target = next(iter(ucmerced_loader))
        print(f"UC Merced Batch - Image shape: {img.shape}, Target: {target.item()}")
        print("UC Merced test passed successfully.\n")
    except Exception as e:
        print(f"UC Merced test failed: {e}\n")
if __name__ == "__main__":
    test_pipeline()
