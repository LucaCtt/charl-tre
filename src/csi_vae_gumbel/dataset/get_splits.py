from pathlib import Path
from string import ascii_uppercase

import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.dataset.dataset import CSIDataset

__DATASET_PARTS = 4


def _split_mats(mats: list[np.ndarray], test_size: float, n_parts: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split each matrix in mats into train and test parts.

    Arguments:
        mats: List of CSI matrices to split.
        test_size: Proportion of the dataset to include in the test split.
        n_parts: Number of parts to split each matrix into

    Returns:
        A tuple containing two lists: train matrices and test matrices.

    """
    train_mats, test_mats = [], []

    for mat in mats:
        # 1. Reshape into (n_parts, samples_per_part, ...)
        # This eliminates the manual start/end indexing logic
        samples_per_part = mat.shape[0] // n_parts
        # We handle potential remainder samples by trimming or using exact multiples
        reshaped = mat[: n_parts * samples_per_part].reshape(n_parts, samples_per_part, *mat.shape[1:])

        # 2. Calculate split point for the inner dimension
        split_idx = int(samples_per_part * (1 - test_size))

        # 3. Vectorized slicing
        # reshaped[:, :split_idx] gives all train parts at once
        train_part = reshaped[:, :split_idx].reshape(-1, *mat.shape[1:])
        test_part = reshaped[:, split_idx:].reshape(-1, *mat.shape[1:])

        train_mats.append(train_part)
        test_mats.append(test_part)

    return train_mats, test_mats


def get_splits(
    dataset_path: Path,
    batch_size: int,
    window_size: int,
    overlap_size: int,
    n_activities: int,
    n_antennas: int,
    antenna_select: int,
    test_size: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Build the CSI dataset train/test dataloaders with DistributedSampler.

    Arguments:
        dataset_path: Path to the dataset directory.
        batch_size: Batch size for the dataloaders.
        window_size: Window size for the CSI samples.
        overlap_size: Overlap size for the CSI samples.
        n_activities: Number of activities (files) to load from the dataset.
        n_antennas: Number of antennas to use from the CSI data.
        antenna_select: Antenna selection strategy.
        test_size: Proportion of the dataset to include in the test split.
        shuffle: Whether to shuffle the data in the dataloaders.

    Returns:
        A tuple containing the training and testing DataLoaders.

    """
    files = [dataset_path / f"S1a_{x}.mat" for x in ascii_uppercase[:n_activities]]
    mats = [np.array(sio.loadmat(file)["csi"]) for file in files]

    train_mats, test_mats = _split_mats(mats, test_size=test_size, n_parts=__DATASET_PARTS)

    # Shape of dataset samples: (n_antennas, window_size, n_subcarriers)
    train_dataset = CSIDataset(
        csi_mats=train_mats,
        window_size=window_size,
        overlap_size=overlap_size,
        n_antennas=n_antennas,
        antenna_select=antenna_select,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # DistributedSampler already shuffles the data
        sampler=DistributedSampler(train_dataset, shuffle=True),  # Shuffle train data
    )

    test_dataset = CSIDataset(
        csi_mats=test_mats,
        window_size=window_size,
        overlap_size=overlap_size,
        n_antennas=n_antennas,
        antenna_select=antenna_select,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(test_dataset, shuffle=False),  # Do not shuffle test data
    )

    return train_dataloader, test_dataloader
