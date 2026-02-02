from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VAEParameters:
    """Parameters for the VAE training."""

    start_learning_rate: float
    final_kl_weight: float
    final_entropy_weight: float
    n_categories: int
    latent_dim: int

    final_capacity: float
    gumbel_temp: float
    loss_type: Literal["bce", "mse"]
