"""Variational Autoencoder (VAE) model definition."""

from typing import Literal

import torch
from torch import nn

from csi_vae_gumbel.model.decoder import CSIDecoder
from csi_vae_gumbel.model.encoder import CSIEncoder


class VAE(nn.Module):
    """Variational Autoencoder for CSI data."""

    def __init__(
        self,
        enc_input_shape: tuple[int, int, int] = (450, 2048, 1),
        dec_input_shape: tuple[int, int, int] = (9, 8, 32),
        latent_dim: int = 2,
        categorical_dim: int = 2,
    ) -> None:
        super().__init__()

        self.encoder = CSIEncoder(enc_input_shape, latent_dim, categorical_dim)
        self.decoder = CSIDecoder(dec_input_shape, latent_dim, categorical_dim, enc_input_shape[-1])

    def forward(
        self,
        x: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE."""
        z, _ = self.encoder(x, tau)
        return self.decoder(z), z


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    z: torch.Tensor,
    kl_weight: float = 5e-4,
    free_bits: float = 0.0,
    entropy_weight: float = 0.0,
    entropy_mode: Literal["none", "penalty", "bonus"] = "none",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss with categorical latent variables.

    Args:
        x_recon (torch.Tensor): Reconstructed input.
        x_true (torch.Tensor): True input.
        z (torch.Tensor): Latent variable tensor.
        kl_weight (float): Weight for the KL divergence term.
        free_bits (float): Free bits threshold for KL divergence.
        prior_prob (float): Prior probability for the categorical distribution.
        entropy_weight (float): Weight for the entropy term.
        entropy_mode (Literal["none", "penalty", "bonus"]): Mode for entropy term.
        eps (float): Small constant for numerical stability.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss,
            KL divergence, entropy, entropy term.

    """
    # Reconstruction loss
    recon = nn.functional.binary_cross_entropy(x_recon, x_true, reduction="mean")

    # Log q(y|x)
    log_z = nn.functional.log_softmax(z, dim=-1)
    log_z = log_z.clamp_min(eps)

    # Posterior q(y|x)
    probs_z = nn.functional.softmax(z, dim=-1)
    posterior_distrib = torch.distributions.Categorical(probs=probs_z)
    posterior = posterior_distrib.probs

    # Prior p(y): uniform categorical
    prior = torch.ones_like(z) / z.size(-1) # z.size(-1) is the categorical dimension
    prior = prior.clamp_min(eps)
    prior_distrib = torch.distributions.Categorical(probs=prior)
    log_prior = prior_distrib.probs.log()

    # KL(q || p)
    kl_per_sample = (posterior * (log_z - log_prior)).view(z.size(0), -1).sum(dim=1)
    if free_bits > 0.0:
        kl_per_sample = nn.functional.relu(kl_per_sample - free_bits)
    kl = kl_per_sample.mean()

    # Entropy H(q): -sum q log q, safe version
    entropy_per_sample = -(posterior * log_z).view(z.size(0), -1).sum(dim=1)
    entropy = entropy_per_sample.mean()

    # Entropy term (sign depends on mode)
    if entropy_mode == "penalty":
        ent_term = +entropy_weight * entropy
    elif entropy_mode == "bonus":
        ent_term = -entropy_weight * entropy
    else:
        ent_term = torch.Tensor([0.0]).to(x_recon.device)

    total_loss = recon + kl_weight * kl + ent_term

    return total_loss, recon, kl, entropy, ent_term
