"""Configuration settings for the model and training."""

from datetime import UTC
from datetime import datetime as dt
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    dataset_path = Path("../dataset/S1")
    """Local path to store the S1 dataset."""
    n_activities = 12
    """Total number of activities in the dataset"""
    n_samples = 12000
    """Number of samples to extract from each CSI matrix file."""
    window_size = 450
    """Size of the sliding window to extract from each sample."""
    n_antennas = 1
    """Total number of antennas used, either a single one or all of them."""
    antenna = 0
    """If n_antennas==1, select which antenna to use (0 to 3). Otherwise, this value is ignored."""

    # Categorical VAE config
    latent_dim = 2
    categorical_dim = n_activities
    vae_name = f"vaed_s1a_a{antenna}_ls{latent_dim}" if n_antennas == 1 else f"vaed_s1a_f_ls{latent_dim}"
    checkpoint_dir = Path(
        f"../vaed_models_{n_activities}activities/{dt.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}/{vae_name}",
    )

    # Training config
    batch_size = 24
    n_epochs = 1
    patience = 3
    learning_rate = 1e-3
