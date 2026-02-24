import torch
from torch import distributions as dist
from torch.nn import functional as func


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    logits: torch.Tensor,
    mus: torch.Tensor,
    logvars: torch.Tensor,
    kl_weight: float,
    capacity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the VAE loss with categorical latent variables.

    Arguments:
        x_recon (torch.Tensor): Reconstructed input.
        x_true (torch.Tensor): True input.
        logits (torch.Tensor): Logits tensor for the categorical latent variables.
        mus (torch.Tensor): Mean tensor for the Gaussian latents.
        logvars (torch.Tensor): Log-variance tensor for the Gaussian latents.
        kl_weight (float): Weight for the KL divergence term.
        capacity (float): Capacity threshold for KL divergence.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss,
            KL divergence.

    """
    # Reconstruction loss
    recon = func.mse_loss(x_recon, x_true, reduction="mean")

    # Categorical KL with capacity control
    num_classes = logits.size(-1)
    q = dist.Categorical(logits=logits)
    p = dist.Categorical(probs=torch.full_like(logits, 1.0 / num_classes))
    kl_cat = dist.kl_divergence(q, p).view(logits.size(0), -1).sum(dim=1).mean()
    if capacity > 0.0:
        kl_cat = torch.clamp(kl_cat - capacity, min=0.0)

    # Gaussian KL for antenna latents
    logvars = logvars.clamp(-10, 10)  # Clamp for numerical stability
    kl_gauss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=(1, 2)).mean()

    total_loss = recon + kl_weight * (kl_cat + 1e-3 * kl_gauss)

    return total_loss, recon, kl_cat
