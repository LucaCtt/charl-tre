from pathlib import Path
from string import ascii_uppercase

import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.dataset.dataset import CSIDataset


def get_splits(
    dataset_path: Path,
    batch_size: int,
    window_size: int,
    overlap_size: int,
    n_activities: int,
    n_samples: int,
    n_antennas: int,
    antenna_select: int,
    test_size: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Build the CSI dataset train/test dataloaders with DistributedSampler."""
    files = [dataset_path / f"S1a_{x}.mat" for x in ascii_uppercase[:n_activities]]
    mats = [np.array(sio.loadmat(file)["csi"]) for file in files]

    # Shape of dataset samples: (n_antennas, window_size, n_subcarriers)
    train_dataset = CSIDataset(
        csi_mats=mats,
        samples_start=0,
        n_samples=int(n_samples * (1 - test_size)),
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
        sampler=DistributedSampler(train_dataset),
    )

    test_dataset = CSIDataset(
        csi_mats=mats,
        samples_start=int(n_samples * (1 - test_size)),
        n_samples=int(n_samples * test_size),
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
        sampler=DistributedSampler(test_dataset),
    )

    return train_dataloader, test_dataloader
