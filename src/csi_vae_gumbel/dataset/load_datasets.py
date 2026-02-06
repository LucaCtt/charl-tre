from pathlib import Path
from string import ascii_uppercase

import numpy as np
import scipy.io as sio

from csi_vae_gumbel.dataset.dataset import CSIDataset

__DATASET_PARTS = 4


def _split_mats(mats: list[np.ndarray], test_ratio: float, n_parts: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split each matrix in mats into train and test parts.

    Arguments:
        mats: List of CSI matrices to split.
        test_ratio: Ratio of the dataset to allocate to the test set.
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
        # Trim to the largest prefix whose length is evenly divisible by n_parts,
        # discarding any leftover samples that would prevent equal-sized parts.
        reshaped = mat[: n_parts * samples_per_part].reshape(n_parts, samples_per_part, *mat.shape[1:])

        # 2. Calculate split point for the inner dimension
        split_idx = int(samples_per_part * (1 - test_ratio))

        # 3. Vectorized slicing
        # reshaped[:, :split_idx] gives all train parts at once
        train_part = reshaped[:, :split_idx].reshape(-1, *mat.shape[1:])
        test_part = reshaped[:, split_idx:].reshape(-1, *mat.shape[1:])

        train_mats.append(train_part)
        test_mats.append(test_part)

    return train_mats, test_mats


def load_datasets(
    dataset_path: Path,
    train_window_size: int,
    overlap_size: int,
    n_activities: int,
    n_antennas: int,
    antenna_select: int,
    test_ratio: float = 0.3,
) -> tuple[CSIDataset, CSIDataset]:
    """Build the CSI train/test datasets.

    Arguments:
        dataset_path: Path to the dataset directory.
        train_window_size: Window size for training samples.
        overlap_size: Overlap size for the CSI samples.
        n_activities: Number of activities (files) to load from the dataset.
        n_antennas: Number of antennas to use from the CSI data.
        antenna_select: Antenna selection strategy.
        test_ratio: Ratio of the dataset to allocate to the test set (default: 0.3).

    Returns:
        A tuple containing the train and test CSIDatasets.

    """
    files = [dataset_path / f"S1a_{x}.mat" for x in ascii_uppercase[:n_activities]]
    mats = [np.array(sio.loadmat(file)["csi"]) for file in files]

    train_mats, test_mats = _split_mats(mats, test_ratio=test_ratio, n_parts=__DATASET_PARTS)

    # Shape of dataset samples: (n_antennas, window_size, n_subcarriers)
    train_dataset = CSIDataset(
        csi_mats=train_mats,
        window_size=train_window_size,
        overlap_size=overlap_size,
        n_antennas=n_antennas,
        antenna_select=antenna_select,
    )

    test_dataset = CSIDataset(
        csi_mats=test_mats,
        window_size=train_window_size,
        overlap_size=0,
        n_antennas=n_antennas,
        antenna_select=antenna_select,
        augment_probability=0.0,
    )

    return train_dataset, test_dataset
