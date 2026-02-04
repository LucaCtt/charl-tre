from dataclasses import dataclass


@dataclass(frozen=True)
class VAEParameters:
    """Parameters for the VAE training."""

    start_lr: float
    final_kl_weight: float
    latent_dim: int
    final_cap: float
    gumbel_temp: float
