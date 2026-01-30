from typing import Literal

import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as func


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    z: torch.Tensor,
    kl_weight: float,
    free_bits: float = 1.0,
    entropy_weight: float = 0.0,
    entropy_mode: Literal["none", "penalty", "bonus"] = "none",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss with categorical latent variables.

    Arguments:
        x_recon (torch.Tensor): Reconstructed input.
        x_true (torch.Tensor): True input.
        z (torch.Tensor): Latent variable tensor.
        kl_weight (float): Weight for the KL divergence term.
        free_bits (float): Free bits threshold for KL divergence.
        entropy_weight (float): Weight for the entropy term.
        entropy_mode (Literal["none", "penalty", "bonus"]): Mode for entropy term.
        eps (float): Small value to avoid numerical issues.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss,
            KL divergence, entropy term.

    """
    # Reconstruction loss
    recon = func.binary_cross_entropy(x_recon.clamp(eps, 1 - eps), x_true, reduction="mean")

    # Posterior q(y|x)
    q = dist.Categorical(logits=z)

    # Prior p(y): uniform categorical
    num_classes = z.size(-1)
    p = dist.Categorical(probs=torch.full_like(z, 1.0 / num_classes))

    # KL(q || p), per sample
    kl_per_sample = dist.kl_divergence(q, p)
    kl_per_sample = kl_per_sample.view(z.size(0), -1).sum(dim=1)
    if free_bits > 0.0:
        kl_per_sample = nn.functional.relu(kl_per_sample - free_bits)
    kl = kl_per_sample.mean()

    # Entropy H(q)
    entropy_per_sample = q.entropy()
    entropy_per_sample = entropy_per_sample.view(z.size(0), -1).sum(dim=1)
    entropy = entropy_per_sample.mean()

    # Entropy term
    if entropy_mode == "penalty":
        ent_term = +entropy_weight * entropy
    elif entropy_mode == "bonus":
        ent_term = -entropy_weight * entropy
    else:
        ent_term = torch.zeros((), device=x_recon.device)

    total_loss = recon + kl_weight * kl + ent_term

    return total_loss, recon, kl, ent_term
