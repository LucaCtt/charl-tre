import math

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    model_config = SettingsConfigDict(env_file=".env")

    # General training config
    debug: bool = False
    """Enable debug mode with more verbose logging and synchronous multi-GPU training."""
    seed: int = 42
    """Random seed for reproducibility."""
    n_epochs: int = 150
    """Number of training epochs for both VAE and classifier."""
    train_batch_size: int = 512 * 3
    """Batch size for training both VAE and classifier. Will be divided by the number of GPUs used."""

    # Dataset config
    dataset_path: str = "dataset/S1"
    """Local path to store the S1 dataset."""
    n_activities: int = 12
    """Total number of activities in the dataset."""
    activities_labels: list[str] = [
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
    n_samples: int = 12000
    """Number of samples to extract from each CSI matrix file."""
    window_size: int = 75
    """Size of the window of CSI in a VAE sample, where 150 CSI = 1 second of data."""
    test_window_factor: int = 6
    """Number of train windows to concatenate for each test sample, to evaluate on longer sequences."""
    test_ratio: float = 0.3
    """Proportion of the dataset to be used for testing."""
    n_antennas: int = 1
    """Total number of antennas used."""
    antenna_select: int = 0
    """Specific antenna to select if only one is needed (0-indexed)."""
    n_subcarriers: int = 2048
    """Number of subcarriers in each CSI sample."""

    # Optuna study config
    n_trials: int = 100
    """Number of Optuna trials for hyperparameter optimization."""
    n_categories: int = math.ceil(math.log2(n_activities))
    """Number of categories for the Gumbel-Softmax distribution."""
    study_name: str = f"a{n_antennas}_w{window_size}_t{n_trials}"
    """Name of the VAE model, used for checkpointing."""
    study_path: str = f"out/{study_name}"
    """Directory to save model checkpoints."""

    # Optuna hyperparameter search space
    latent_dim_min: int = 2
    """Minimum latent dimension size."""
    latent_dim_max: int = 8
    """Maximum latent dimension size."""
    final_kl_weight_min: float = 5e-3
    """Minimum final KL divergence weight in loss computation."""
    final_kl_weight_max: float = 5e-1
    """Maximum final KL divergence weight in loss computation."""
    final_cap_min: float = 0.5
    """Minimum final capacity in loss computation."""
    final_cap_max: float = 4
    """Maximum final capacity in loss computation."""
    start_gumbel_temp_min: float = 2
    """Minimum Gumbel-Softmax temperature."""
    start_gumbel_temp_max: float = 5
    """Maximum Gumbel-Softmax temperature."""
