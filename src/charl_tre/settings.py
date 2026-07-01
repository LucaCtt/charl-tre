from typing import Literal, NamedTuple

from pydantic_settings import BaseSettings, SettingsConfigDict

from charl_tre.models.vae.dirichlet import CONV_SPECS


class ParamRange[NumberT](NamedTuple):
    """Defines a hyperparameter with a name and a range."""

    min: NumberT
    max: NumberT
    step: NumberT | None | Literal["log"] = None


class ParamCategorical[T](NamedTuple):
    """Defines a hyperparameter with a name and a list of values."""

    values: list[T]


class Settings(BaseSettings):
    """Configuration settings for the model and training."""

    model_config = SettingsConfigDict(env_file=".env")

    # General settings
    seed: int = 42
    """Random seed for reproducibility."""
    num_workers: int = 8

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
    train_window_size: int = 75
    """Size of the window of CSI in a VAE train sample, where 150 CSI = 1 second of data."""
    test_window_size: int = 450
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
    early_stop_patience: int = 20
    """Number of epochs to wait before early stopping."""
    early_stop_warmup_epochs: int = 10
    """Number of epochs to wait before starting to check for early stopping."""
    free_bits_start: float  = 1.0
    """Initial free-bits floor for the KL divergence term."""
    free_bits_end: float = 0.05
    """Final free-bits floor for the KL divergence term."""

    # Optuna study settings
    n_trials: int = 100
    """Number of Optuna trials for hyperparameter optimization."""
    study_name: str = f"a{n_antennas}_w{train_window_size}_tw{test_window_size}"
    """Name of the VAE model, used for checkpointing."""
    study_path: str = f"out/{study_name}"
    """Directory to save model checkpoints."""
    collapse_patience: int = 5
    """Number of epochs to wait before collapsing the latent space."""

    # Optuna hyperparameter search space
    batch_size: ParamRange[int] = ParamRange(min=32, max=128, step=32)
    """Batch size per GPU for training and testing both VAE and classifier."""
    lr: ParamRange[float] = ParamRange(min=1e-3, max=3e-2, step="log")
    """Learning rate for the optimizer."""
    kl_max: ParamRange[float] = ParamRange(min=0.5, max=4.0, step=0.5)
    """Maximum weight for the KL divergence term during annealing."""
    n_components: ParamRange[int] = ParamRange(min=8, max=64, step=8)
    """Number of components in the Dirichlet distribution (latent space dimensionality)."""
    conv_layers_spec: ParamCategorical[int] = ParamCategorical(values=[*range(len(CONV_SPECS))])
    """Convolutional layers specification for the VAE encoder."""
    prior_alpha: ParamRange[float] = ParamRange(min=0.1, max=10.0, step="log")
    """Prior alpha for the Dirichlet distribution."""
    n_fusion_layers: ParamRange[int] = ParamRange(min=1, max=3, step=1)
    """Number of layers in the delayed fusion classifier."""
    fusion_dropout: ParamRange[float] = ParamRange(min=0.0, max=0.3, step=0.1)
    """Dropout rate to use in the delayed fusion classifier."""

    @property
    def n_activities(self) -> int:
        """Return the number of activities based on the length of the activities list."""
        return len(self.activities)

    @property
    def n_train_windows_in_test(self) -> int:
        """Return the number of train windows that compose the test window, counting the overlap."""
        return (self.test_window_size - self.train_window_size) // self.stride + 1

    @property
    def overlap_size(self) -> int:
        """Return the number of overlapping frames between consecutive test windows."""
        return self.train_window_size - self.stride
