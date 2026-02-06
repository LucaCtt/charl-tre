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
    train_batch_size: int = 32 * 3
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
    train_window_size: int = 150
    """Size of the window of CSI in a VAE train sample, where 150 CSI = 1 second of data."""
    overlap_size: int = 15
    """Size of the overlap between two consecutive windows."""
    n_antennas: int = 1
    """Total number of antennas used."""
    antenna_select: int = 0
    """Specific antenna to select if only one is needed (0-indexed)."""
    n_subcarriers: int = 2048
    """Number of subcarriers in each CSI sample."""

    # Optuna study config
    study_name: str = f"vae_gumbel_a{n_antennas}"
    """Name of the VAE model, used for checkpointing."""
    study_dir: str = f"out/{study_name}"
    """Directory to save model checkpoints."""
    n_trials: int = 100
    """Number of Optuna trials for hyperparameter optimization."""
    n_categories: int = math.ceil(math.log2(n_activities))

    # Optuna hyperparameter search space
    latent_dim_min: int = 1
    """Minimum latent dimension size."""
    latent_dim_max: int = 10
    """Maximum latent dimension size."""
    start_lr_min: float = 1e-4
    """Minimum starting learning rate."""
    start_lr_max: float = 3e-3
    """Maximum starting learning rate."""
    final_kl_weight_min: float = 1e-5
    """Minimum final KL divergence weight in loss computation."""
    final_kl_weight_max: float = 1e-2
    """Maximum final KL divergence weight in loss computation."""
    final_cap_min: float = 1e-1
    """Minimum final capacity in loss computation."""
    final_cap_max: float = 2
    """Maximum final capacity in loss computation."""
    gumbel_temp_min: float = 1e-2
    """Minimum Gumbel-Softmax temperature."""
    gumbel_temp_max: float = 2
    """Maximum Gumbel-Softmax temperature."""
