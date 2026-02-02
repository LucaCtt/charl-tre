from datetime import UTC
from datetime import datetime as dt

from numpy import log2
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    model_config = SettingsConfigDict(env_file=".env")

    # General training config
    debug: bool = True
    """Enable debug mode with more verbose logging and synchronous multi-GPU training."""
    seed: int = 42
    """Random seed for reproducibility."""
    n_trials: int = 30
    """Number of Optuna trials for hyperparameter optimization."""
    n_epochs: int = 50
    """Number of training epochs for each Optuna trial, for both VAE and classifier."""
    batch_size: int = 12 * 3
    """Batch size for training both VAE and classifier."""

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
    window_size: int = 150
    """Size of the window of CSI in a sample, where 150 CSI = 1 second of data."""
    overlap_size: int = 15
    """Size of the overlap between two consecutive windows."""
    n_antennas: int = 1
    """Total number of antennas used."""
    antenna_select: int = 0
    """Specific antenna to select if only one is needed (0-indexed)."""
    n_subcarriers: int = 2048
    """Number of subcarriers in each CSI sample."""

    # Checkpointing config
    vae_name: str = f"vaec_s1a_ls{n_activities}"
    """Name of the VAE model, used for checkpointing."""
    checkpoint_dir: str = f"out/vaec_models/{dt.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}/{vae_name}"
    """Directory to save model checkpoints."""

    # Optuna hyperparameter search space
    n_cats_min: int = int(log2(n_activities))
    n_cats_max: int = int(log2(n_activities) * 4)
    latent_dim_min: int = 1
    latent_dim_max: int = 16
    start_lr_min: float = 1e-4
    start_lr_max: float = 3e-3
    final_entr_weight_min: float = 1e-5
    final_entr_weight_max: float = 1e-2
    final_kl_weight_min: float = 1e-5
    final_kl_weight_max: float = 1e-2
    final_cap_min: float = 1e-1
    final_cap_max: float = 2
    gumbel_temp_min: float = 1e-2
    gumbel_temp_max: float = 2
