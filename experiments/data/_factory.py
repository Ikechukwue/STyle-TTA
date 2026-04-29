"""
xAILab Bamberg
University of Bamberg

@description:
Dataset factory.
"""

# Import packages
import os
from torchvision.datasets.vision import VisionDataset
from typing import Callable, Optional

# Import own scripts
from experiments.data.camelyon import Camelyon17WILDS
from experiments.data.epistr import EpitheliumStroma
from experiments.data.peripheral_blood import PeripheralBlood
from experiments.data.bone_marrow_smears_and_peripheral_blood import BoneMarrowSmearsAndPeripheralBlood
from experiments.data.fitzpatrick import Fitzpatrick17k
from experiments.data.ddi import DDI
from experiments.data.retina import Retina
#from experiments.data.medmnist_dataset import MedMNIST, MEDMNIST_2D_DATASETS
from experiments.data.imagefolder import ImageFolder
from experiments.data.cifar import CIFAR
from experiments.data.imagenet_variants import ImageNet
from experiments.data.pacs import PACS
from experiments.data.vlcs import VLCS
from experiments.data.domainnet import DomainNet
from experiments.data.office_home import OfficeHome
from experiments.data.office_caltech import OfficeCaltech
from experiments.data.visda17 import VisDA17
from experiments.data.visual_decathlon import VisualDecathlon
from experiments.data.terra_incognita import TerraIncognita
from experiments.data.colored_mnist import ColoredMNIST
from experiments.data.nico import NICO
from experiments.data.metashift import MetaShift
from experiments.data.openmibood import OpenMIBOOD


class CustomDataset(VisionDataset):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs
    ):
        # Extract subset parameters
        use_subset = kwargs.get('use_subset', False)
        subset_size = kwargs.get('subset_size', None)
        subset_seed = kwargs.get('subset_seed', None)

        # Load the dataset
        if dataset_name.lower() == "camelyon17wilds":
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, dataset_name)

            # Load the data with the default dataset splits
            self.dataset = Camelyon17WILDS(root_dir=root_dir, split=split,
                                           transform=transform,
                                           target_transform=target_transform,
                                           use_subset=use_subset,
                                           subset_size=subset_size,
                                           subset_seed=subset_seed)

        elif dataset_name.lower() == "epistr":
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, dataset_name)

            # Load the data with dataset splits across datasets (NKI, VGH, IHC)
            # Train: NKI / Val: VGH / Test: IHC
            self.dataset = EpitheliumStroma(root_dir=root_dir, split=split,
                                            transform=transform,
                                            target_transform=target_transform,
                                            use_subset=use_subset,
                                            subset_size=subset_size,
                                            subset_seed=subset_seed)

        elif dataset_name.lower() == "peripheral_blood":
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, dataset_name)

            # Load the Peripheral Blood dataset
            # Train: Matek19 (80%) + Acevedo20 (80%)
            # Val: Matek19 (20%) + Acevedo20 (20%)
            # Test: MLL23
            self.dataset = PeripheralBlood(root_dir=root_dir, split=split,
                                           transform=transform,
                                           target_transform=target_transform,
                                           use_subset=use_subset,
                                           subset_size=subset_size,
                                           subset_seed=subset_seed)

        elif dataset_name.lower() == "bone_marrow_smears_and_peripheral_blood":
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, dataset_name)

            # Load the Bone Marrow Smears and Peripheral Blood dataset
            # Train: BMC
            # Val: Matek19
            # Test: MLL23
            self.dataset = BoneMarrowSmearsAndPeripheralBlood(root_dir=root_dir,
                                                              split=split,
                                                              transform=transform,
                                                              target_transform=target_transform,
                                                              use_subset=use_subset,
                                                              subset_size=subset_size,
                                                              subset_seed=subset_seed)

        elif "fitzpatrick17k" in dataset_name.lower():
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, "fitzpatrick17k")

            # Load the data with dataset splits across skin tones (1,2,3,4,5,6)
            # Default: Train: 1,2 / Val: 3,4 / Test: 5,6
            self.dataset = Fitzpatrick17k(root_dir=root_dir, split=split,
                                          transform=transform,
                                          target_transform=target_transform,
                                          train_categories=kwargs.get('train_categories', (1, 2)),
                                          val_categories=kwargs.get('val_categories', (3, 4)),
                                          test_categories=kwargs.get('test_categories', (5, 6)),
                                          use_subset=use_subset,
                                          subset_size=subset_size,
                                          subset_seed=subset_seed)

        elif "ddi" in dataset_name.lower():
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, "ddi")

            # Load the data with dataset splits across skin tones (12,34,56)
            # Default: Train: 12 / Val: 34 / Test: 56
            self.dataset = DDI(root_dir=root_dir, split=split, transform=transform,
                               target_transform=target_transform,
                               train_categories=kwargs.get('train_categories', (12,)),
                               val_categories=kwargs.get('val_categories', (34,)),
                               test_categories=kwargs.get('test_categories', (56,)),
                               use_subset=use_subset,
                               subset_size=subset_size,
                               subset_seed=subset_seed)

        elif dataset_name.lower() == "retina":
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, dataset_name)

            # Load the Retina dataset
            # Train: APTOS + DeepDR / Val: IDRiD / Test: MESSIDOR-2
            self.dataset = Retina(root_dir=root_dir, split=split,
                                  transform=transform,
                                  target_transform=target_transform,
                                  use_subset=use_subset,
                                  subset_size=subset_size,
                                  subset_seed=subset_seed)

        elif dataset_name.lower() == "imagefolder":
            # Load images from a specified folder path
            # data_path should be the direct path to the image folder
            self.dataset = ImageFolder(root=root_dir, transform=transform,
                                       target_transform=target_transform,
                                       use_subset=use_subset,
                                       subset_size=subset_size,
                                       subset_seed=subset_seed)
            
            """
            elif dataset_name.lower() in MEDMNIST_2D_DATASETS:
                # Create the root directory to the data if it does not exist
                root_dir = create_dataset_directory(data_path, "medmnist")

                # Load the MedMNIST dataset with default 224x224 size
                size = kwargs.get('size', 224)  # Default to 224x224
                self.dataset = MedMNIST(
                    root_dir=root_dir,
                    dataset_name=dataset_name,
                    split=split,
                    size=size,
                    transform=transform,
                    target_transform=target_transform,
                    use_subset=use_subset,
                    subset_size=subset_size,
                    subset_seed=subset_seed,
                    download=True
                )
            """

        elif dataset_name.lower() in ("cifar10", "cifar100"):
            root_dir = create_dataset_directory(data_path, "cifar")
            num_classes = 10 if dataset_name.lower() == "cifar10" else 100
            self.dataset = CIFAR(
                root_dir=root_dir, split=split, num_classes=num_classes,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "imagenet":
            root_dir = create_dataset_directory(data_path, "imagenet")
            self.dataset = ImageNet(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "pacs":
            root_dir = create_dataset_directory(data_path, "pacs")
            self.dataset = PACS(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "vlcs":
            root_dir = create_dataset_directory(data_path, "vlcs")
            self.dataset = VLCS(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "domainnet":
            root_dir = create_dataset_directory(data_path, "domainnet")
            self.dataset = DomainNet(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "office_home":
            root_dir = create_dataset_directory(data_path, "office_home")
            self.dataset = OfficeHome(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "office_caltech":
            root_dir = create_dataset_directory(data_path, "office_caltech")
            self.dataset = OfficeCaltech(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "visda17":
            root_dir = create_dataset_directory(data_path, "visda17")
            self.dataset = VisDA17(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "visual_decathlon":
            root_dir = create_dataset_directory(data_path, "visual_decathlon")
            self.dataset = VisualDecathlon(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "terra_incognita":
            root_dir = create_dataset_directory(data_path, "terra_incognita")
            self.dataset = TerraIncognita(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "colored_mnist":
            root_dir = create_dataset_directory(data_path, "colored_mnist")
            self.dataset = ColoredMNIST(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "nico":
            root_dir = create_dataset_directory(data_path, "nico")
            self.dataset = NICO(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "metashift":
            root_dir = create_dataset_directory(data_path, "metashift")
            self.dataset = MetaShift(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        elif dataset_name.lower() == "openmibood":
            root_dir = create_dataset_directory(data_path, "openmibood")
            self.dataset = OpenMIBOOD(
                root_dir=root_dir, split=split,
                transform=transform, target_transform=target_transform,
                use_subset=use_subset, subset_size=subset_size,
                subset_seed=subset_seed, **kwargs)

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_dataset_directory(data_path: str, dataset_name: str) -> str:
    """
    Create a directory for the dataset.

    Args:
        data_path (str): Path to store/load the dataset.
        dataset_name (str): Name of the dataset.

    Returns:
        str: Path to the dataset directory.
    """

    # Create the root directory to the data if it does not exist
    root_dir = os.path.join(data_path, dataset_name)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    return root_dir


def create_dataset(dataset_name: str, data_path: str, split: str = "train",
                   transform: Optional[Callable] = None,
                   target_transform: Optional[Callable] = None,
                   **kwargs) -> CustomDataset:
    """
    Create a dataset with specified split using the CustomDataset class.

    Args:
        dataset_name (str): Name of the dataset.
        data_path (str): Path to store/load the dataset.
        split (str): Which split to load ('train', 'val', 'test').
        transform (callable, optional): A function/transform to apply to the data.
        target_transform (callable, optional): A function/transform to apply
            to the target.
        kwargs: Additional keyword arguments for the dataset.

    Returns:
        dataset: The requested dataset split or all splits as a dictionary.
    """

    return CustomDataset(dataset_name, data_path, split, transform, target_transform,
                         **kwargs)
