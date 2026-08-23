import os
import random

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def init_rng(seed: int) -> None:
    """Initialize random seeds for reproducibility.

    Arguments:
        seed (int): The random seed to use for all random number generators.

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dl(ds: Dataset, batch_size: int, shuffle: bool, n_workers: int, seed: int) -> DataLoader:
    """Create a DataLoader with common settings.

    Handles distributed training by using a DistributedSampler when torch.distributed is initialized.

    Arguments:
        ds: Dataset to load data from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the beginning of each epoch.
        n_workers: Number of subprocesses to use for data loading.
        seed: Random seed for reproducibility of shuffling.

    Returns:
        A DataLoader instance for the given dataset and settings.

    """
    is_ddp_spawned = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    n_workers = 0 if is_ddp_spawned else n_workers

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if not is_ddp_spawned else False,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
        sampler=DistributedSampler(ds, shuffle=shuffle, seed=seed) if is_ddp_spawned else None,
    )
