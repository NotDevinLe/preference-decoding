import matplotlib.pyplot as plt
import matplotlib
import os
from functools import partial
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from math import inf
import pandas as pd
import torchvision.transforms as transforms
import yaml
import numpy as np
import fs_datasets

matplotlib.use("Agg")


def _sample_binarize(x):
    return torch.distributions.Bernoulli(probs=x).sample()


def _binarize(x, threshold=0.5):
    x[x >= threshold] = 1
    x[x < threshold] = 0
    return x


TRANSFORMS_DICT = {
    "mnist": {
        "train": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.1306], std=[0.3080])]
        ),
        "test": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.1306], std=[0.3080])]
        ),
    },
    "mnist_fashion": {
        "train": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.2873], std=[0.3529])]
        ),
        "test": lambda _: transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.2873], std=[0.3529])]
        ),
    },
    "rewards": {
        "train": lambda _: None,  # No transforms needed for reward data
        "test": lambda _: None,
    },
}


def load_custom_dataset(train_path, val_path, split="train"):
    """
    Load custom data from specified file paths.
    The data should be in format nxd where n is training samples and d is attributes.
    
    Args:
        train_path: Full path to training data file
        val_path: Full path to validation data file
        split: "train" or "valid"
    """
    import torch
    from torch.utils.data import TensorDataset
    
    if split == "train":
        file_path = train_path
    elif split == "valid":
        file_path = val_path
    else:
        raise ValueError(f"Invalid split: {split}. Only 'train' and 'valid' are supported.")
        
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
        
    # Load data based on file extension
    if file_path.endswith('.pt'):
        data = torch.load(file_path)
    elif file_path.endswith('.npy'):
        import numpy as np
        data = torch.from_numpy(np.load(file_path)).float()
    elif file_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(file_path, header=None)
        data = torch.from_numpy(df.values).float()
    else:
        raise ValueError(f"Unsupported file format. Supported: .pt, .npy, .csv")
        
    print(f"Loaded {split} data from {os.path.basename(file_path)} with shape: {data.shape}")
    
    # Create dummy labels (since this is unsupervised learning)
    n_samples, n_attributes = data.shape
    dummy_labels = torch.zeros(n_samples, 1)  # Dummy labels for compatibility
    
    return TensorDataset(data, dummy_labels)


def load_rewards_dataset(root, split="train"):
    """
    Load reward data from data/rewards.pt file.
    The data is in format nxd where n is training samples and d is attributes.
    matrix[i][j] is the reward signal for example i under attribute j.
    """
    rewards_path = os.path.join(root, "rewards.pt")
    return load_custom_dataset(rewards_path, split)


def get_dataset(dataset_str, root, input_size=None, train_path=None, val_path=None):
    transform_train = None
    transform_val = None
    
    # Handle custom rewards dataset
    if dataset_str == "rewards":
        dataset_train = load_rewards_dataset(root, "train")
        dataset_val = load_rewards_dataset(root, "valid")
        return (dataset_train, dataset_val)
    
    # Handle custom dataset with separate train/val files
    if dataset_str == "custom":
        if train_path is None or val_path is None:
            raise ValueError("Both train_path and val_path must be specified when using 'custom' dataset")
        
        dataset_train = load_custom_dataset(train_path, val_path, "train")
        dataset_val = load_custom_dataset(train_path, val_path, "valid")
        return (dataset_train, dataset_val)
    
    if dataset_str == "mnist" or dataset_str == "mnist_fashion":
        transform_train = TRANSFORMS_DICT[dataset_str]["train"](input_size)
        transform_val = TRANSFORMS_DICT[dataset_str]["test"](input_size)
        transform_test = TRANSFORMS_DICT[dataset_str]["test"](input_size)

    dataset_train = fs_datasets.get_dataset_split(
        dataset_str, root, "train", transform_train
    )
    dataset_val = fs_datasets.get_dataset_split(
        dataset_str, root, "valid", transform_val
    )
    dataset_test = fs_datasets.get_dataset_split(
        dataset_str, root, "test", transform_test
    )
    return (dataset_train, dataset_val, dataset_test)


def get_dataset_mean_std(dataset, num_workers=0):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
    from tqdm import tqdm

    probe_im, _ = next(iter(dataloader))
    channels = probe_im.shape[1]
    mins = torch.tensor([torch.inf] * channels)
    maxes = torch.tensor([-torch.inf] * channels)
    sum_ = 0
    sq_sum = 0
    count = 0
    for im, _ in tqdm(dataloader):
        # im = im.squeeze(0) numbers
        dims = im.shape
        sum_ += torch.sum(im, dim=(0, 2, 3))
        sq_sum += torch.sum(torch.pow(im, 2), dim=(0, 2, 3))
        count += im[:, 0, :].flatten().shape[0]
        mins = torch.minimum(mins, torch.amin(im, dim=(0, 2, 3)))
        maxes = torch.maximum(maxes, torch.amax(im, dim=(0, 2, 3)))
    print(dims)
    mean = sum_ / count
    sq_mean = sq_sum / count
    std = torch.sqrt(sq_mean - torch.pow(mean, 2))
    return mean, std, (maxes, mins)


def get_num_parameters(model):
    sum = 0
    for param in list(model.parameters()):
        sum += param.numel()
    return sum


def get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def create_sparse_grid(side_length=28, k=50):
    # Determine the dimensions of the grid
    n_rows = int(np.ceil(np.sqrt(k)))
    n_cols = int(np.ceil(k / n_rows))

    # Initialize an empty image
    image = np.zeros((side_length, side_length), dtype=int)

    # Calculate spacing based on the more filled dimension
    if n_cols > n_rows:
        spacing = side_length / n_cols
        n_filled_rows = min(n_rows, int(np.ceil(k / n_cols)))
    else:
        spacing = side_length / n_rows
        n_filled_rows = n_rows

    # Calculate offsets to center the grid
    offset_x = (side_length - (spacing * n_cols)) / 2
    offset_y = (side_length - (spacing * n_filled_rows)) / 2

    # Populate the grid
    for i in range(k):
        row = int(i / n_cols)
        col = i % n_cols
        x = int(offset_y + row * spacing)
        y = int(offset_x + col * spacing)

        if x < side_length and y < side_length:
            image[x, y] = 1

    return image
