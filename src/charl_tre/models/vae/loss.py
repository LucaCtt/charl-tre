"""Loss functions for Dirichlet VAE."""

import torch
import torch.nn.functional as func


def dirichlet_kl_divergence(
    alpha: torch.Tensor,
    prior_alpha: float = 1.0,
) -> torch.Tensor:
    """Compute KL divergence between Dirichlet distributions.

    KL(q(z|x) || p(z)) where q(z|x) = Dir(alpha) and p(z) = Dir(prior_alpha)

    Arguments:
        alpha: Concentration parameters of shape (batch_size, n_components)
        prior_alpha: Concentration parameter for the symmetric Dirichlet prior (default: 1.0)
                    If 1.0, this is a uniform prior

    Returns:
        kl_loss: KL divergence averaged over batch of shape (batch_size,)

    """
    # Prior concentration parameters
    prior_alpha_tensor = torch.full_like(alpha, prior_alpha)

    # Compute log of Gamma functions
    log_gamma_alpha = torch.lgamma(alpha)
    log_gamma_sum_alpha = torch.lgamma(alpha.sum(dim=-1))

    log_gamma_prior = torch.lgamma(prior_alpha_tensor)
    log_gamma_sum_prior = torch.lgamma(prior_alpha_tensor.sum(dim=-1))

    # Digamma function (derivative of log Gamma)
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(alpha.sum(dim=-1, keepdim=True))

    # KL divergence formula
    # KL = log B(prior) - log B(alpha) + sum((alpha - prior) * (digamma(alpha) - digamma(sum(alpha))))
    # where B(alpha) = Gamma(sum(alpha)) / prod(Gamma(alpha))

    log_beta_prior = log_gamma_sum_prior - log_gamma_prior.sum(dim=-1)
    log_beta_alpha = log_gamma_sum_alpha - log_gamma_alpha.sum(dim=-1)

    kl = log_beta_prior - log_beta_alpha
    kl += ((alpha - prior_alpha_tensor) * (digamma_alpha - digamma_sum_alpha)).sum(dim=-1)

    return kl


def dirichlet_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    alpha: torch.Tensor,
    kl_weight: float = 1.0,
    prior_alpha: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the ELBO loss for Dirichlet VAE.

    Loss = Reconstruction Loss + KL Weight * KL Divergence

    Arguments:
        x_recon: Reconstructed input of shape (batch_size, ...)
        x_true: Ground truth input of shape (batch_size, ...)
        alpha: Concentration parameters of shape (batch_size, n_components)
        kl_weight: Weight for KL divergence term (default: 1.0)
        prior_alpha: Concentration parameter for the symmetric Dirichlet prior (default: 1.0)
        reduction: How to reduce the loss ('mean' or 'sum')

    Returns:
        total_loss: Combined ELBO loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence loss

    """
    # Reconstruction loss (Mean Squared Error)
    recon_loss = func.mse_loss(x_recon, x_true, reduction=reduction)

    # KL divergence loss
    kl_per_sample = dirichlet_kl_divergence(alpha, prior_alpha)
    if reduction == "mean":
        kl_loss = kl_per_sample.mean()
    elif reduction == "sum":
        kl_loss = kl_per_sample.sum()
    else:
        msg = f"Invalid reduction method: {reduction}"
        raise ValueError(msg)

    # Total loss (ELBO)
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss

