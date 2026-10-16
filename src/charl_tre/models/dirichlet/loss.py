import torch
import torch.nn.functional as func


def loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    alpha: torch.Tensor,
    kl_weight: float = 1.0,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate reconstruction and Dirichlet KL losses.

    Args:
            x_recon: Reconstructed input.
            x_true: Original input.
            alpha: Dirichlet concentration parameters of shape (batch_size, n_components).
            kl_weight: Weight applied to the Dirichlet KL divergence.
            free_bits: Minimum KL contribution per sample.

    Returns:
            A tuple containing total loss, reconstruction loss, and Dirichlet KL loss.

    """
    recon_loss = func.mse_loss(x_recon, x_true)
    kl_samples = kl_divergence(alpha).clamp(min=free_bits)
    kl_loss = kl_samples.mean()
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


def kl_divergence(alpha: torch.Tensor) -> torch.Tensor:
    """Calculate the KL divergence between a Dirichlet distribution and a uniform prior.

    Arguments:
        alpha (torch.Tensor): The concentration parameters of the Dirichlet distribution.

    Returns:
        torch.Tensor: The KL divergence for each sample in the batch.

    """
    prior_alpha = torch.full_like(alpha, 1)

    # Log gammas for numerical stability
    log_gamma_alpha = torch.lgamma(alpha)
    log_gamma_sum_alpha = torch.lgamma(alpha.sum(dim=-1))
    log_gamma_prior = torch.lgamma(prior_alpha)
    log_gamma_sum_prior = torch.lgamma(prior_alpha.sum(dim=-1))

    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(alpha.sum(dim=-1, keepdim=True))

    log_b_prior = log_gamma_prior.sum(dim=-1) - log_gamma_sum_prior
    log_b_alpha = log_gamma_alpha.sum(dim=-1) - log_gamma_sum_alpha

    kl = log_b_prior - log_b_alpha
    kl += ((alpha - prior_alpha) * (digamma_alpha - digamma_sum_alpha)).sum(dim=-1)
    return kl
