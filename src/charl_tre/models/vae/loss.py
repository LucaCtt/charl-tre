"""Loss functions for Dirichlet VAE."""

import torch
import torch.nn.functional as func


def dirichlet_kl_divergence(alpha: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """Compute KL divergence between Dirichlet distributions.

    KL(q(z|x) || p(z)) where q(z|x) = Dir(alpha) and p(z) = Dir(1)

    Arguments:
        alpha: Concentration parameters of shape (batch_size, n_components)
        free_bits: Number of free bits to allow in the KL divergence (default: 0.0)

    Returns:
        kl_loss: KL divergence averaged over batch of shape (batch_size,)

    """
    # Prior concentration parameters
    prior_alpha_tensor = torch.full_like(alpha, 1)

    # Compute log of Gamma functions
    log_gamma_alpha = torch.lgamma(alpha)
    log_gamma_sum_alpha = torch.lgamma(alpha.sum(dim=-1))

    log_gamma_prior = torch.lgamma(prior_alpha_tensor)
    log_gamma_sum_prior = torch.lgamma(prior_alpha_tensor.sum(dim=-1))

    # Digamma function (derivative of log Gamma)
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(alpha.sum(dim=-1, keepdim=True))

    # KL(Dir(alpha) || Dir(prior)) = log B(prior)
    # - log B(alpha) + sum((alpha - prior) * (digamma(alpha) - digamma(sum(alpha))))
    # Standard: log B(x) = sum(lgamma(x_k)) - lgamma(sum(x_k))
    log_b_prior = log_gamma_prior.sum(dim=-1) - log_gamma_sum_prior
    log_b_alpha = log_gamma_alpha.sum(dim=-1) - log_gamma_sum_alpha

    kl = log_b_prior - log_b_alpha
    kl += ((alpha - prior_alpha_tensor) * (digamma_alpha - digamma_sum_alpha)).sum(dim=-1)

    # Apply free bits constraint
    return torch.clamp(kl, min=free_bits)


def dirichlet_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    alpha: torch.Tensor,
    kl_weight: float = 1.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the ELBO loss for Dirichlet VAE.

    Loss = Reconstruction Loss + KL Weight * KL Divergence

    Arguments:
        x_recon: Reconstructed input of shape (batch_size, ...)
        x_true: Ground truth input of shape (batch_size, ...)
        alpha: Concentration parameters of shape (batch_size, n_components)
        kl_weight: Weight for KL divergence term (default: 1.0)
        free_bits: Number of free bits to allow in the KL divergence (default: 0.0)

    Returns:
        total_loss: Combined ELBO loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss

    """
    # Reconstruction loss (Mean Squared Error)
    recon_loss = func.mse_loss(x_recon, x_true)

    # KL divergence loss
    kl_per_sample = dirichlet_kl_divergence(alpha, free_bits)
    kl_loss = kl_per_sample.mean()

    # Total loss (ELBO)
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss
