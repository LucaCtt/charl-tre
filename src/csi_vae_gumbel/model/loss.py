import math
from typing import Literal

import torch
from torch import nn


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    z: torch.Tensor,
    kl_weight: float = 1e-3,
    free_bits: float = 1.0,
    entropy_weight: float = 0.0,
    entropy_mode: Literal["none", "penalty", "bonus"] = "none",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss with categorical latent variables.

    Arguments:
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

    # Posterior q(y|x): categorical, softmax over logits
    posterior = nn.functional.softmax(z, dim=-1)

    # Prior p(y): uniform categorical, where p(y) = 1/K
    # Skip straight to log p(y)
    log_prior = torch.full_like(z, -math.log(z.size(-1)))

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
        ent_term = torch.zeros((), device=x_recon.device)

    total_loss = recon + kl_weight * kl + ent_term

    return total_loss, recon, kl, entropy, ent_term
