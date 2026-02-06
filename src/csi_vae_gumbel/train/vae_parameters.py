from dataclasses import dataclass


@dataclass(frozen=True)
class VAEParameters:
    """Parameters for the VAE training."""

    final_kl_weight: float
    latent_dim: int
    final_cap: float
    start_gumbel_temp: float
