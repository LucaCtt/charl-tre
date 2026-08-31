import numpy as np
import torch
from torch.utils.data import Dataset


class MultiAntenna(Dataset):
    """CSI dataset with lazy sliding-window indexing over multiple antennas."""

    def __init__(
        self,
        csi_mats: list[np.ndarray],
        window_size: int,
        stride: int,
    ) -> None:
        """Initialize the MultiAntenna dataset.

        Arguments:
            csi_mats: List of CSI matrices, one per activity, each with shape
                (n_antennas, n_parts, n_samples, n_subcarriers).
            window_size: Number of time steps per window.
            stride: Step size between consecutive windows.

        Raises:
            ValueError: If any matrix has fewer time steps than window_size,
                or if csi_mats is empty.

        """
        if not csi_mats:
            msg = "csi_mats must not be empty"
            raise ValueError(msg)

        if window_size < 1 or stride < 1:
            msg = f"window_size and stride must be >= 1, got {window_size=}, {stride=}"
            raise ValueError(msg)

        self.__window_size = window_size

        self.__data: list[np.ndarray] = []
        self.__index_map: list[tuple[int, int, int]] = []

        # Load files once, build index map
        for label, csi in enumerate(csi_mats):
            _, n_parts, n_samples, _ = csi.shape
            if n_parts * n_samples < window_size:
                msg = f"Window size {window_size} is larger than the total number of samples for label {label}"
                raise ValueError(msg)

            if (n_samples - window_size) % stride != 0:
                msg = (
                    f"Window size {window_size} and stride {stride} would create a window that is not fully contained "
                    f"in the partition for label {label}. Consider adjusting the window size or stride."
                )
                raise ValueError(msg)

            self.__data.append(csi)

            # Build lazy sliding-window index
            for part in range(n_parts):
                for offset in range(0, n_samples - window_size + 1, stride):
                    self.__index_map.append((label, part, offset))

    def __len__(self) -> int:
        return len(self.__index_map)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        label, part, start = self.__index_map[index]
        csi = self.__data[label]

        window = csi[:, part, start : start + self.__window_size, :]

        x = torch.from_numpy(window)
        return x, label
