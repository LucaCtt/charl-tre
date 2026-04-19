import numpy as np
import torch
from torch.utils.data import Dataset

from charl_tre.dataset.augmenter import CSIAugmenter


class CSIDataset(Dataset):
    """CSI Dataset for PyTorch.

    Shape of dataset items is [n_antennas, window_size, n_subcarriers]
    """

    def __init__(
        self,
        csi_mats: list[np.ndarray],
        window_size: int,
        n_antennas: int,
        antenna_select: int,
        stride: int,
        seed: int,
        augment_probability: float = 0.3,
        normalize: bool = True,
    ) -> None:
        """Initialize the CSI dataset.

        Arguments:
            csi_mats: List of CSI matrices loaded from .mat files. Each matrix should have shape
                [num_samples, n_subcarriers, n_antennas].
            window_size: Size of the sliding window to extract from each sample.
            n_antennas: Total number of antennas used, either a single one or all of them.
            antenna_select: Specific antenna to select if only one is needed. If None, use all antennas.
            stride: Stride of the sliding window.
            seed: Random seed for reproducibility of data augmentation.
            augment_probability: Probability of applying data augmentation to the CSI data.
            normalize: Whether to normalize the CSI data by the global maximum value.

        """
        self.__window_size = window_size
        self.__augmenter = CSIAugmenter(seed=seed)
        self.__normalize = normalize
        self.__augment_probability = augment_probability

        self.__data: list[np.ndarray] = []
        self.__labels: list[int] = []
        self.__index_map: list[tuple[int, int]] = []

        self.__global_min = float("inf")
        self.__global_max = 0.0

        # Load files once, build index map
        for label, csi_mat in enumerate(csi_mats):
            # 802.11ax has 2048 subcarriers (160 MHz bandwidth), we can keep one data
            # point every 4 subcarriers to reduce input size, make the methodology compatible
            # with 802.11ac (popular in literature) while still keeping most of the information.
            # Shape of csi for now is: [num_samples, n_subcarriers, n_antennas]
            # We will later rearrange it.
            csi = csi_mat[:, ::4, :]

            # We can further discard the second half of the subcarriers
            # and keep most of the information,
            csi = csi[:, : csi.shape[1] // 2, :]

            if n_antennas == 1:
                csi = csi[..., antenna_select]
                csi = csi[:, :, np.newaxis]  # Keep 3D shape for consistency

            # Discard phase information, keep only magnitude.
            # Phase is often very noisy and not very informative.
            csi = np.round(np.abs(csi)).astype(np.float32)

            # Get global max for normalization if needed
            if self.__normalize:
                self.__global_max = max(self.__global_max, np.max(csi))
                self.__global_min = min(self.__global_min, np.min(csi))

            file_id = len(self.__data)
            self.__data.append(csi)
            self.__labels.append(label)

            # Build lazy sliding-window index
            for start in range(0, csi.shape[0] - window_size + 1, stride):
                self.__index_map.append((file_id, start))

    def __len__(self) -> int:
        return len(self.__index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_id, start = self.__index_map[idx]
        csi = self.__data[file_id]

        window = csi[start : start + self.__window_size]

        # (window_size, n_subcarriers, n_antennas) → (n_antennas, window_size, n_subcarriers)
        # The window_size represents the time dimension
        window = np.transpose(window, (2, 0, 1))

        # Normalize if needed
        if self.__normalize:
            window = (window - self.__global_min) / (self.__global_max - self.__global_min + 1e-12)

        # Apply augmentation if needed
        if torch.rand(1).item() < self.__augment_probability:
            window = self.__augmenter.apply(window.copy())

        x = torch.from_numpy(window)
        y = self.__labels[file_id]

        return x, y
