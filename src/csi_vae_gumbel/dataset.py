from pathlib import Path
from string import ascii_uppercase

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class CSIDataset(Dataset):
    """CSI Dataset for PyTorch.

    Shape of dataset items is [n_antennas, window_size, n_subcarriers]
    """

    def __init__(
        self,
        files: list[Path],
        n_samples: int,
        window_size: int,
        overlap_size: int,
        n_antennas: int,
        normalize: bool = True,
    ) -> None:
        """Initialize the CSI dataset.

        Arguments:
            files: List of paths to the .mat files containing the CSI data.
            n_samples: Number of samples to extract from each CSI matrix file.
            window_size: Size of the sliding window to extract from each sample.
            overlap_size: Size of the overlap between two consecutive windows.
            downsample_factor: Factor by which to downsample the window size.
            n_antennas: Total number of antennas used, either a single one or all of them.
            normalize: Whether to normalize the CSI data by the global maximum value.

        """
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.n_antennas = n_antennas
        self.normalize = normalize

        self.data = []
        self.labels = []
        self.index_map = []

        # Load files once, build index map
        for label, file in enumerate(files):
            # num_samples, n_subcarriers, n_antennas
            mat = sio.loadmat(file)

            # Shape of csi for now is: [num_samples, n_subcarriers, n_antennas]
            # We will later rearrange it.
            csi = np.array(mat["csi"])
            csi = csi[:n_samples, ..., :n_antennas]

            # 802.11ax has 2048 subcarriers (160 MHz bandwidth), we can keep one data
            # point every 4 subcarriers to reduce input size, make the methodology compatible
            # with 802.11ac (popular in literature) while still keeping most of the information.
            csi = csi[:, ::4, :]

            # We can further discard the second half of the subcarriers
            # and keep most of the information,
            csi = csi[:, : csi.shape[1] // 2, :]

            # Discard phase information, keep only magnitude.
            # Phase is often very noisy and not very informative.
            csi = np.round(np.abs(csi)).astype(np.float32)

            file_id = len(self.data)
            self.data.append(csi)
            self.labels.append(label)

            # Build lazy sliding-window index
            step_size = window_size - overlap_size
            for start in range(0, n_samples - window_size, step_size):
                self.index_map.append((file_id, start))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_id, start = self.index_map[idx]
        csi = self.data[file_id]

        window = csi[start : start + self.window_size]

        # (window_size, n_subcarriers, n_antennas) → (n_antennas, window_size, n_subcarriers)
        # The window_size represents the time dimension
        window = np.transpose(window, (2, 0, 1))

        # Min-max normalization per window, not global to avoid outliers issues
        if self.normalize:
            win_min = window.min()
            win_max = window.max()
            window = (window - win_min) / (win_max - win_min + 1e-8)

        x = torch.from_numpy(window)
        y = self.labels[file_id]

        return x, y


def get_splits(
    dataset_path: Path,
    batch_size: int,
    window_size: int,
    overlap_size: int,
    n_activities: int,
    n_samples: int,
    n_antennas: int,
    test_size: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Build the CSI dataset train/test dataloaders with DistributedSampler."""
    files = [dataset_path / f"S1a_{x}.mat" for x in ascii_uppercase[:n_activities]]

    # Shape of dataset samples: (n_antennas, window_size, n_subcarriers)
    dataset = CSIDataset(
        files=files,
        n_samples=n_samples,
        window_size=window_size,
        overlap_size=overlap_size,
        n_antennas=n_antennas,
    )

    n_total = len(dataset)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [n_train, n_test],
    )

    # Split train data over gpus
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # DistributedSampler already shuffles the data
        sampler=DistributedSampler(train_dataset),
    )

    # Keep test data on cpu, no need to distribute it
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    return train_dataloader, test_dataloader
