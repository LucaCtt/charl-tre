import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as func


def vae_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    logits: torch.Tensor,
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
        entropy_weight (float): Weight for the entropy term.
        entropy_mode (Literal["none", "penalty", "bonus"]): Mode for entropy term.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Total loss, reconstruction loss,
            KL divergence.

    """
    # Reconstruction loss
    recon = func.binary_cross_entropy_with_logits(x_recon, x_true, reduction="mean")

    # Posterior q(y|x)
    q = dist.Categorical(logits=logits)

    # Prior p(y): uniform categorical
    num_classes = logits.size(-1)
    p = dist.Categorical(probs=torch.full_like(logits, 1.0 / num_classes))

    # KL(q || p), per sample
    kl_per_sample = dist.kl_divergence(q, p)
    kl_per_sample = kl_per_sample.view(logits.size(0), -1).sum(dim=1)
    if capacity > 0.0:
        kl_per_sample = nn.functional.relu(kl_per_sample - capacity)
    kl = kl_per_sample.mean()

    total_loss = recon + kl_weight * kl

    return total_loss, recon, kl
