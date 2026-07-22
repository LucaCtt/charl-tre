from pathlib import Path

import numpy as np


class LatentsLoader:
    """Load latents from disk and split them by activity label.

    It is assumed that the latents are already saved in time-lagged format,
    with shape (n_windows, n_mixtures, n_components).
    The corresponding activity labels should be saved in a separate file with shape (n_windows,).
    """

    def __init__(self, study_path: str | Path) -> None:
        """Initialize with base path to study data."""
        self.__base = Path(study_path) / "latents"

    def load(
        self,
        test_ratio: float = 0.3,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Load latents and labels from disk.

        Arguments:
            test_ratio (float): Fraction of data to use for testing.

        Returns:
            tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
                - Dictionary mapping activity labels to training latents.
                - Dictionary mapping activity labels to testing latents.

        """
        alpha = np.load(self.__base / "alpha.npy")
        labels = np.load(self.__base / "labels.npy").astype(int)

        _, _, n_components = alpha.shape
        latents = np.eye(n_components, dtype=np.int8)[np.argmax(alpha, axis=2)]

        by_activity = {int(label): latents[labels == label] for label in np.unique(labels)}

        train: dict[int, np.ndarray] = {}
        test: dict[int, np.ndarray] = {}

        for label, data in by_activity.items():
            n = len(data)

            split_idx = int(n * (1 - test_ratio))
            split_idx = max(1, min(split_idx, n - 1))

            train[label] = data[:split_idx]
            test[label] = data[split_idx:]

        return train, test
