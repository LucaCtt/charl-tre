import torch
from torch.nn import functional as func


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
