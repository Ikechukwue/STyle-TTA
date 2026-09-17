"""
xAILab Bamberg
University of Bamberg

@description:
Common utility functions used by the dataset classes.

@author: Sebastian Doerrich
"""

# Import packages
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


def create_own_train_val_subset(dataset: Dataset, split: str = "train",
                                size_val_set: float = 0.1,
                                seed_train_val_split: int = 0) -> Subset:
    """
    Split dataset into own train and val subsets.

    Args:
        dataset (Dataset): The dataset to split into train and val
        split (str): Which split to load ('train' or 'val')
        size_val_set (float): The size of the validation set given in proportion
            to the training set
        seed_train_val_split (int): The seed for the train-val split

    Returns:
        (Subset): The requested train or val split of the dataset
    """

    # Check the input
    assert isinstance(dataset, Dataset), "The dataset must be a PyTorch Dataset."
    assert isinstance(split, str), "The split must be a string."
    assert isinstance(size_val_set, float), "The size of the validation set must be "
    "a float."
    assert isinstance(seed_train_val_split, int), "The seed for the train-val split "
    "must be an integer."
    assert split in ["train", "val"], "The split must be either 'train' or 'val'."
    assert 0 < size_val_set < 1, "The size of the validation set must be between "
    "0 and 1."

    # Extract labels for stratification
    labels = [label for _, label in dataset]

    # Split the dataset into training and validation
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=size_val_set,
        random_state=seed_train_val_split,
        stratify=labels,
    )

    # Return the requested split
    if split == "train":
        return Subset(dataset, train_idx)

    else:
        return Subset(dataset, val_idx)


def create_own_train_val_test_subset(dataset: Dataset, split: str = "train",
                                     size_test_set: float = 0.2,
                                     size_val_set: float = 0.1,
                                     seed_train_val_test_split: int = 0) -> Subset:
    """
    Split dataset into own train, val and test subsets.

    Args:
        dataset (Dataset): The dataset to split into train and val
        split (str): Which split to load ('train' or 'val')
        size_test_set (float): The size of the test set given in proportion to
            the entire data set
        size_val_set (float): The size of the validation set given in proportion to
            the training set
        seed_train_val_test_split (int): The seed for the train-val-test split

    Returns:
        (Subset): The requested train, val or test split of the dataset
    """

    # Check the input
    assert isinstance(dataset, Dataset), "The dataset must be a PyTorch Dataset."
    assert isinstance(split, str), "The split must be a string."
    assert isinstance(size_test_set, float), "The size of the test set must be a float."
    assert isinstance(size_val_set, float), "The size of the validation set must be "
    "a float."
    assert isinstance(seed_train_val_test_split, int), "The seed for the "
    "train-val-test split must be an integer."
    assert split in ["train", "val", "test"], "The split must be either 'train', 'val' "
    "or 'test'."
    assert 0 < size_test_set < 1, "The size of the test set must be between 0 and 1."
    assert 0 < size_val_set < 1, "The size of the validation set must be between "
    "0 and 1."

    # Extract labels for stratification
    labels = [label for _, label in dataset]

    # Split the dataset into training+validation and test
    train_val_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=size_test_set,
        random_state=seed_train_val_test_split,
        stratify=labels,
    )

    # Extract labels for the training+validation set
    train_val_labels = [labels[i] for i in train_val_idx]

    # Split the training+validation set into training and validation
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=size_val_set,
        random_state=seed_train_val_test_split,
        stratify=train_val_labels,
    )

    # Create the training, validation, and test splits
    if split == "train":
        return Subset(dataset, train_idx)

    elif split == "val":
        return Subset(dataset, val_idx)

    else:
        return Subset(dataset, test_idx)
