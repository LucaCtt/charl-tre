import torch
from torch import distributions as dist
from torch.nn import functional as func


def categorical_vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    logits: torch.Tensor,
    latents_pre: torch.Tensor,
    latents_post: torch.Tensor,
    kl_weight: float,
    capacity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss with categorical latent variables.

    Arguments:
        x_recon (torch.Tensor): Reconstructed input.
        x_true (torch.Tensor): True input.
        logits (torch.Tensor): Logits tensor for the categorical latent variables.
        kl_weight (float): Weight for the KL divergence term.
        capacity (float): Capacity threshold for KL divergence.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss,
            KL divergence.

    """
    # Reconstruction loss
    per_ant_mse = torch.mean((x_recon - x_true) ** 2, dim=(0, 2, 3))
    recon = per_ant_mse.sum() + per_ant_mse.max()  # Penalize the "weakest link" antenna

    feature_recon = func.mse_loss(latents_post, latents_pre, reduction="mean")

    # Categorical KL with capacity control
    num_classes = logits.size(-1)
    q = dist.Categorical(logits=logits)
    p = dist.Categorical(probs=torch.full_like(logits, 1.0 / num_classes))
    kl = dist.kl_divergence(q, p).view(logits.size(0), -1).sum(dim=1).mean()
    if capacity > 0.0:
        kl = torch.clamp(kl - capacity, min=0.0)

    total_loss = recon + 0.5 * feature_recon + kl_weight * kl

    return total_loss, recon, kl


def gaussian_vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss with Gaussian latent variables.

    Arguments:
        x_recon (torch.Tensor): Reconstructed input.
        x_true (torch.Tensor): True input.
        mu (torch.Tensor): Mean of the latent distribution.
        logvar (torch.Tensor): Log variance of the latent distribution.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss,
            KL divergence.

    """
    # Reconstruction loss
    recon = func.mse_loss(x_recon, x_true, reduction="mean")

    # KL divergence with standard normal prior
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon + kl

    return total_loss, recon, kl
