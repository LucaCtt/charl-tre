from datetime import UTC
from datetime import datetime as dt

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    model_config = SettingsConfigDict(env_file=".env")

    debug: bool = True

    # Dataset config
    dataset_path: str = "dataset/S1"
    """Local path to store the S1 dataset."""
    n_activities: int = 12
    """Total number of activities in the dataset."""
    n_samples: int = 12000
    """Number of samples to extract from each CSI matrix file."""
    window_size: int = 150
    """Size of the sliding window to extract from each sample."""
    n_antennas: int = 4
    """Total number of antennas used."""
    n_subcarriers: int = 2048
    """Number of subcarriers in each CSI sample."""
    hidden_latent_dim: int = 128
    """Dimension of the hidden latent space per antenna."""

    # Categorical VAE config
    latent_dim: int = 2
    categorical_dim: int = n_activities
    vae_name: str = f"vaed_s1a_ls{latent_dim * categorical_dim}"
    checkpoint_dir: str = f"out/vaed_models/{dt.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}/{vae_name}"

    # Training config
    batch_size: int = 8 * 3
    n_epochs: int = 50
    patience: int = 3
    learning_rate: float = 1e-3
