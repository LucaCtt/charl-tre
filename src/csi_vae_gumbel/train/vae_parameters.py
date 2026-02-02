from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VAEParameters:
    """Parameters for the VAE training."""

    start_lr: float
    final_kl_weight: float
    final_entr_weight: float
    n_cats: int
    latent_dim: int
    final_cap: float
    gumbel_temp: float
    loss_type: Literal["bce", "mse"]
