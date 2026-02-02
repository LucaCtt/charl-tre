from datetime import UTC
from datetime import datetime as dt

from numpy import log2
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    model_config = SettingsConfigDict(env_file=".env")

    debug: bool = False

    n_trials: int = 10
    """Number of Optuna trials for hyperparameter optimization."""

    # Dataset config
    dataset_path: str = "dataset/S1"
    """Local path to store the S1 dataset."""
    n_activities: int = 12
    """Total number of activities in the dataset."""
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
    batch_size: int = 12 * 3

    # Categorical VAE config
    vae_name: str = f"vaec_s1a_ls{n_activities}"
    checkpoint_dir: str = f"out/vaec_models/{dt.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}/{vae_name}"

    # Hyperparameters
    n_epochs: int = 50
    min_n_categories: int = int(log2(n_activities))
    max_n_categories: int = int(log2(n_activities) * 4)
    min_latent_dim: int = 1
    max_latent_dim: int = 16
    min_learning_rate: float = 1e-4
    max_learning_rate: float = 3e-3
    min_entropy_weight: float = 1e-5
    max_entropy_weight: float = 1e-2
    min_kl_weight: float = 1e-5
    max_kl_weight: float = 1e-2
    min_final_capacity: float = 1e-1
    max_final_capacity: float = 2
    min_gumbel_temp: float = 1e-2
    max_gumbel_temp: float = 2
