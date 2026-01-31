import numpy as np
from scipy.interpolate import interp1d


class CSIAugmenter:
    """Provides various augmentation strategies for CSI data tensors."""

    def __init__(
        self,
        noise_std: float = 0.01,
        mask_prob: float = 0.1,
        antenna_drop_prob: float = 0.1,
        apply_prob: float = 0.5,
        augmentation_prob: float = 0.3,
    ) -> None:
        """Initialize the augmenter with specified probabilities and parameters.

        Arguments:
            noise_std: Standard deviation of Gaussian noise to add.
            mask_prob: Probability of masking each subcarrier.
            antenna_drop_prob: Probability of dropping an entire antenna's signal.
            apply_prob: Overall probability of applying augmentations to a sample.
            augmentation_prob: Probability of applying each individual augmentation.

        """
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.antenna_drop_prob = antenna_drop_prob
        self.apply_prob = apply_prob
        self.augmentation_prob = augmentation_prob
        self.rng = np.random.default_rng()

    def add_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """Simulate thermal noise in the WiFi hardware."""
        noise = self.rng.normal(0, self.noise_std, x.shape)
        noisy = x + noise
        return np.clip(noisy, 0.0, 1.0).astype(np.float32)

    def subcarrier_masking(self, x: np.ndarray) -> np.ndarray:
        """Randomly masks frequency subcarriers (columns) to simulate interference."""
        mask = self.rng.random(x.shape[-1]) < self.mask_prob
        # Broadcast mask across antennas and time
        return x * mask[np.newaxis, np.newaxis, :]

    def antenna_dropout(self, x: np.ndarray) -> np.ndarray:
        """Randomly zeroes out an entire antenna's signal (spatial shadowing)."""
        if x.shape[0] > 1 and self.rng.random() < self.antenna_drop_prob:
            ant_idx = self.rng.integers(0, x.shape[0])
            x[ant_idx, :, :] = 0
        return x

    def time_warp(self, x: np.ndarray) -> np.ndarray:
        """Randomly stretches or compresses the time dimension."""
        c, t, s = x.shape
        # Create a random warp factor between 0.8 and 1.2
        warp_factor = self.rng.uniform(0.8, 1.2)
        new_t = int(t * warp_factor)

        # Interpolate across the time axis (axis 1)
        x_indices = np.linspace(0, t - 1, t)
        new_indices = np.linspace(0, t - 1, new_t)

        # We need to warp each antenna and subcarrier
        warped_x = np.zeros((c, t, s), dtype=np.float32)
        for ci in range(c):
            f = interp1d(x_indices, x[ci], axis=0, kind="linear", fill_value="extrapolate")  # type: ignore[call-arg]
            temp = f(new_indices)
            # Crop or pad to maintain original window_size (T)
            if new_t > t:
                warped_x[ci] = temp[:t, :]
            else:
                warped_x[ci, :new_t, :] = temp

        return warped_x

    def temporal_block_out(self, x: np.ndarray) -> np.ndarray:
        """Zeroes out a random chunk of time to simulate signal loss."""
        _, t, _ = x.shape
        if t <= 1:
            return x
        # Choose a block size up to 20% of the window (at least 1)
        max_block = max(1, t // 5)
        block_size = int(self.rng.integers(1, max_block + 1))
        start = int(self.rng.integers(0, t - block_size + 1))

        x[:, start : start + block_size, :] = 0
        return x

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply a random suite of augmentations to the input window."""
        if self.rng.random() > self.apply_prob:
            return x

        # Randomly apply a selection of transforms
        if self.rng.random() < self.augmentation_prob:
            x = self.add_gaussian_noise(x)
        if self.rng.random() < self.augmentation_prob:
            x = self.subcarrier_masking(x)
        if self.rng.random() < self.augmentation_prob:
            x = self.antenna_dropout(x)
        if self.rng.random() < self.augmentation_prob:
            x = self.time_warp(x)
        if self.rng.random() < self.augmentation_prob:
            x = self.temporal_block_out(x)

        return x.astype(np.float32)
