"""CSI Dataset module."""

from pathlib import Path
from string import ascii_uppercase

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class CSIDataset(Dataset):
    """CSI Dataset for PyTorch.

    Shape of dataset items is (n_antennas, window_size, n_subcarriers)
    """

    def __init__(
        self,
        files: list[Path],
        n_samples: int,
        window_size: int,
        n_antennas: int,
        antenna: int,
        normalize: bool = True,
    ) -> None:
        self.window_size = window_size
        self.n_antennas = n_antennas
        self.normalize = normalize

        self.data = []
        self.labels = []
        self.index_map = []

        global_max = 0.0

        # Load files once, build index map
        for label, file in enumerate(files):
            # num_samples, n_subcarriers, n_antennas
            mat = sio.loadmat(file)

            csi = np.array(mat["csi"])
            csi = csi[:n_samples, ..., int(antenna)] if n_antennas == 1 else csi[:n_samples]
            csi = np.round(np.abs(csi)).astype(np.float32)

            if normalize:
                global_max = max(global_max, csi.max())

            file_id = len(self.data)
            self.data.append(csi)
            self.labels.append(label)

            # Build lazy sliding-window index
            for start in range(n_samples - window_size):
                self.index_map.append((file_id, start))

        self.global_max = global_max if normalize else 1.0

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_id, start = self.index_map[idx]
        csi = self.data[file_id]

        window = csi[start : start + self.window_size]

        # (window_size, n_subcarriers, n_antennas) → (n_antennas, window_size, n_subcarriers)
        # The window_size represents the time dimension
        window = window[np.newaxis, ...] if self.n_antennas == 1 else np.transpose(window, (2, 0, 1))

        x = torch.from_numpy(window) / self.global_max
        y = self.labels[file_id]

        return x, y


def build_dataloader(
    dataset_path: Path,
    batch_size: int,
    window_size: int,
    n_activities: int,
    n_samples: int,
    n_antennas: int,
    antenna: int,
) -> DataLoader:
    """Build the CSI dataset dataloader with DistributedSampler."""
    files = [dataset_path / f"S1a_{x}.mat" for x in ascii_uppercase[:n_activities]]
    print(f"Found {len(files)} matrix files: {[f.name for f in files]}")

    # Shape of dataset samples: (n_antennas, window_size, n_subcarriers)
    dataset = CSIDataset(
        files=files,
        n_samples=n_samples,
        window_size=window_size,
        n_antennas=n_antennas,
        antenna=antenna,
    )

    # Shape of dataloader batches: (batch_size, n_antennas, window_size, n_subcarriers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,  # DistributedSampler already shuffles the data
        sampler=DistributedSampler(dataset),
    )
