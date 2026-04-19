"""Latent data loading and activity-wise splitting."""

from pathlib import Path

import numpy as np


class LatentDataLoader:
    """Load hard latents from disk and split them by activity label."""

    def __init__(self, study_path: str | Path) -> None:
        """Initialize with base path to study data."""
        self.__base = Path(study_path) / "latents"

    def load(
        self,
        use_full_onehot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, int, int, int]:
        """Load hard latents and labels from disk.

        Arguments:
            use_full_onehot: If True, return latents as one-hot vectors. If False, return latents as integer indices.

        Returns:
            latents: The loaded latents, either as one-hot vectors or integer indices.
            labels: The corresponding activity labels for each latent vector.
            latent_dim: The dimensionality of the latent space (number of latent factors).
            n_categories: The number of categories for each latent factor.

        """
        latents_hard = np.load(self.__base / "latents_hard.npy")
        labels = np.load(self.__base / "labels.npy").astype(int)

        latent_dim = int(latents_hard.shape[1])
        n_categories = int(latents_hard.shape[2])

        if use_full_onehot:
            latents = latents_hard.reshape(latents_hard.shape[0], latent_dim * n_categories).astype(np.int8)
            n_states = 2
        else:
            latents = np.argmax(latents_hard, axis=2).astype(np.int16)
            n_states = n_categories

        return latents, labels, latent_dim, n_categories, n_states

    @staticmethod
    def split_by_activity(latents: np.ndarray, labels: np.ndarray, stride: int) -> dict[int, np.ndarray]:
        """Create one latent time-series per activity label."""
        if stride < 1:
            msg = "stride must be >= 1."
            raise ValueError(msg)
        return {int(label): latents[labels == label][::stride] for label in np.unique(labels)}

    @staticmethod
    def train_test_split(
        labelled_data: dict[int, np.ndarray],
        train_ratio: float,
        min_test_length: int,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Deterministic chronological train/test split per activity."""
        train: dict[int, np.ndarray] = {}
        test: dict[int, np.ndarray] = {}

        for label, data in labelled_data.items():
            n = len(data)

            split_idx = int(n * train_ratio)
            split_idx = max(1, min(split_idx, n - 1))
            if n - split_idx < min_test_length:
                split_idx = max(1, n - min_test_length)

            train[label] = data[:split_idx]
            test[label] = data[split_idx:]

        return train, test
