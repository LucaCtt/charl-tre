from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from csi_vae_gumbel.dataset.augmenter import CSIAugmenter


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
        antenna_select: int,
        augmenter: CSIAugmenter,
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
            antenna_select: Specific antenna to select if only one is needed. If None, use all antennas.
            augmenter: CSIAugmenter instance for data augmentation.
            normalize: Whether to normalize the CSI data by the global maximum value.

        """
        self.__window_size = window_size
        self.__augmenter = augmenter
        self.__normalize = normalize

        self.__data = []
        self.__labels = []
        self.__index_map = []

        # Load files once, build index map
        for label, file in enumerate(files):
            # num_samples, n_subcarriers, n_antennas
            mat = sio.loadmat(file)

            # Shape of csi for now is: [num_samples, n_subcarriers, n_antennas]
            # We will later rearrange it.
            csi = np.array(mat["csi"])
            csi = csi[:n_samples, ..., antenna_select] if n_antennas == 1 else csi[:n_samples, ...]

            if n_antennas == 1:
                csi = csi[:, :, np.newaxis]  # Keep 3D shape for consistency

            # 802.11ax has 2048 subcarriers (160 MHz bandwidth), we can keep one data
            # point every 4 subcarriers to reduce input size, make the methodology compatible
            # with 802.11ac (popular in literature) while still keeping most of the information.
            csi = csi[:, ::4, :]

            # We can further discard the second half of the subcarriers
            # and keep most of the information,
            csi = csi[:, : csi.shape[1] // 2]

            # Discard phase information, keep only magnitude.
            # Phase is often very noisy and not very informative.
            csi = np.round(np.abs(csi)).astype(np.float32)

            file_id = len(self.__data)
            self.__data.append(csi)
            self.__labels.append(label)

            # Build lazy sliding-window index
            step_size = window_size - overlap_size
            for start in range(0, n_samples - window_size, step_size):
                self.__index_map.append((file_id, start, False))
                self.__index_map.append((file_id, start, True))  # Augmented version

    def __len__(self) -> int:
        return len(self.__index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_id, start, augmented = self.__index_map[idx]
        csi = self.__data[file_id]

        window = csi[start : start + self.__window_size]

        # (window_size, n_subcarriers, n_antennas) → (n_antennas, window_size, n_subcarriers)
        # The window_size represents the time dimension
        window = np.transpose(window, (2, 0, 1))

        # Apply augmentation if needed
        if augmented:
            window = self.__augmenter.apply(window.copy())

        # Min-max normalization per window, not global to avoid outliers issues
        if self.__normalize:
            win_min = window.min()
            win_max = window.max()
            window = (window - win_min) / (win_max - win_min + 1e-8)

        x = torch.from_numpy(window)
        y = self.__labels[file_id]

        return x, y
