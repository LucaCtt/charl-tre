import math

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    model_config = SettingsConfigDict(env_file=".env")

    # General settings
    debug: bool = False
    """Enable debug mode with more verbose logging and synchronous multi-GPU training."""
    seed: int = 42
    """Random seed for reproducibility."""

    # Data and train settings
    dataset_path: str = "dataset/S1"
    """Local path to store the S1 dataset."""
    activities: list[str] = [
        "Walk",
        "Run",
        "Jump",
        "Sit",
        "Empty",
        "Stand",
        "Waving",
        "Clap",
        "Lay down",
        "Wipe",
        "Squat",
        "Stretch",
    ]
    train_window_size: int = 75
    """Size of the window of CSI in a VAE train sample, where 150 CSI = 1 second of data."""
    test_window_size: int = 450
    """Size of the window of CSI in a classifier test sample, where 150 CSI = 1 second of data."""
    test_ratio: float = 0.3
    """Proportion of the dataset to be used for testing."""
    test_overlap_size: int = 50
    """Overlap size between test windows."""
    stride: int = 1
    """Stride for sliding windows creation, for both train and testing."""
    n_antennas: int = 1
    """Total number of antennas used."""
    antenna_select: int = 0
    """Specific antenna to select, used only if n_antennas is 1."""
    n_subcarriers: int = 2048 // 8  # 2048 is original size, downsampled by 8
    """Number of subcarriers in each CSI sample."""
    latent_dim: int = 128
    """Dimensionality of the latent space for each antenna in the VAE."""
    batch_size: int = 144
    """Batch size per GPU for training and testing both VAE and classifier."""
    vae_n_epochs: int = 100
    """Number of training epochs for the VAE."""
    classifier_n_epochs: int = 50
    """Number of training epochs for the classifier."""

    # Optuna study settings
    n_trials: int = 100
    """Number of Optuna trials for hyperparameter optimization."""
    n_categories: int = math.ceil(math.log2(len(activities)))
    """Number of categories for the Gumbel-Softmax distribution."""
    study_name: str = f"a{n_antennas}_w{train_window_size}_tw{test_window_size}_b{batch_size}_s{stride}"
    """Name of the VAE model, used for checkpointing."""
    study_path: str = f"out/{study_name}"
    """Directory to save model checkpoints."""

    # Optuna hyperparameter search space
    latent_dim_min: int = 1
    """Minimum latent dimension size."""
    latent_dim_max: int = 5
    """Maximum latent dimension size."""
    final_kl_weight_min: float = 5e-4
    """Minimum final KL divergence weight in loss computation."""
    final_kl_weight_max: float = 5e-3
    """Maximum final KL divergence weight in loss computation."""
    final_cap_min: float = 0.5
    """Minimum final capacity in loss computation."""
    final_cap_max: float = 2
    """Maximum final capacity in loss computation."""
    start_gumbel_temp_min: float = 2
    """Minimum Gumbel-Softmax temperature."""
    start_gumbel_temp_max: float = 3
    """Maximum Gumbel-Softmax temperature."""

    @property
    def n_activities(self) -> int:
        """Return the number of activities based on the length of the activities list."""
        return len(self.activities)

    @property
    def n_train_windows_in_test(self) -> int:
        """Return the number of train windows that compose the test window, counting the overlap."""
        # We start with one full window, then slide it by (train_window_size - test_overlap_size)
        # until we reach the end of the test window. The integer division counts how many
        # additional shifts fit into the remaining length (test_window_size - train_window_size).
        return 1 + (self.test_window_size - self.train_window_size) // (self.train_window_size - self.test_overlap_size)
