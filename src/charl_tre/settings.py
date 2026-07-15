from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    model_config = SettingsConfigDict(env_file=".env")

    # General settings
    seed: int = 42
    """Random seed for reproducibility."""
    num_workers: int = 8
    """Number of workers for data loading."""

    # Data and train settings
    dataset_path: str = "dataset/S1a.h5"
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
    vae_window_size: int = 75
    """Size of the window of CSI in a VAE train sample, where 150 CSI = 1 second of data."""
    classifier_window_size: int = 450
    """Size of the window of CSI in a classifier test sample, where 150 CSI = 1 second of data."""
    stride: int = 25
    """Stride for sliding windows creation, for both train and testing."""
    n_antennas: int = 4
    """Total number of antennas used."""
    antenna_select: int = 0
    """Specific antenna to select, used only if n_antennas is 1."""
    n_subcarriers: int = 2048 // 8  # 2048 is original size, downsampled by 8
    """Number of subcarriers in each CSI sample."""
    n_epochs: int = 150
    """Number of training epochs for the VAE."""
    early_stop_patience: int = 10
    """Number of epochs to wait before early stopping."""
    early_stop_warmup_epochs: int = 10
    """Number of epochs to wait before starting to check for early stopping."""
    free_bits_start: float = 1.0
    """Initial free-bits floor for the KL divergence term."""
    free_bits_end: float = 0.05
    """Final free-bits floor for the KL divergence term."""

    # Optuna study settings
    n_trials: int = 100
    """Number of Optuna trials for hyperparameter optimization."""
    study_name: str = f"a{n_antennas}_w{vae_window_size}_tw{classifier_window_size}"
    """Name of the VAE model, used for checkpointing."""
    study_path: str = f"out/{study_name}"
    """Directory to save model checkpoints."""
    collapse_patience: int = 5
    """Number of epochs to wait before collapsing the latent space."""
    n_components_penalty_weight: float = 1e-4
    """Weight for the penalty term on the number of components in the loss function."""

    # Optuna hyperparameter search space
    hyperparam_batch_size_min: int = 32
    hyperparam_batch_size_max: int = 96
    hyperparam_batch_size_step: Literal["log"] | int = 32

    hyperparam_lr_min: float = 2e-3
    hyperparam_lr_max: float = 3e-2
    hyperparam_lr_step: Literal["log"] | int = "log"

    hyperparam_kl_final_min: float = 1.5
    hyperparam_kl_final_max: float = 4.0
    hyperparam_kl_final_step: Literal["log"] | float = 0.5

    hyperparam_n_components_min: int = 8
    hyperparam_n_components_max: int = 32
    hyperparam_n_components_step: Literal["log"] | int = 8

    hyperparam_n_mixtures_min: int = 1
    hyperparam_n_mixtures_max: int = 8
    hyperparam_n_mixtures_step: Literal["log"] | int = 1

    hyperparam_n_fusion_layers_min: int = 1
    hyperparam_n_fusion_layers_max: int = 3
    hyperparam_n_fusion_layers_step: Literal["log"] | int = 1

    hyperparam_fusion_dropout_min: float = 0.0
    hyperparam_fusion_dropout_max: float = 0.3
    hyperparam_fusion_dropout_step: Literal["log"] | float = 0.1

    @property
    def n_activities(self) -> int:
        """Return the number of activities based on the length of the activities list."""
        return len(self.activities)

    @property
    def overlap_size(self) -> int:
        """Return the number of overlapping frames between consecutive test windows."""
        return self.vae_window_size - self.stride
