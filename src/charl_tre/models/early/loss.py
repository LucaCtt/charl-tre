import torch
import torch.nn.functional as func


def hierarchical_loss(
    x_recon: torch.Tensor,
    x_true: torch.Tensor,
    mix_logits: torch.Tensor,
    alpha: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
    kl_weight_dirichlet: float = 1.0,
    kl_weight_gaussian: float = 1.0,
    free_bits_dirichlet: float = 0.0,
    free_bits_gaussian: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the hierarchical loss for the model.

    Arguments:
        x_recon (torch.Tensor): The reconstructed input.
        x_true (torch.Tensor): The true input.
        mix_logits (torch.Tensor): The logits for the mixture components.
        alpha (torch.Tensor): The concentration parameters for the Dirichlet distribution.
        mu_q (torch.Tensor): The mean of the posterior distribution.
        logvar_q (torch.Tensor): The log variance of the posterior distribution.
        mu_prior (torch.Tensor): The mean of the prior distribution.
        logvar_prior (torch.Tensor): The log variance of the prior distribution.
        kl_weight_dirichlet (float): The weight for the Dirichlet KL divergence.
        kl_weight_gaussian (float): The weight for the Gaussian KL divergence.
        free_bits_dirichlet (float): The number of free bits for the Dirichlet KL divergence.
        free_bits_gaussian (float): The number of free bits for the Gaussian KL divergence.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: The total loss combining reconstruction and KL divergences.
            - recon_loss: The reconstruction loss.
            - kl_mdm_loss: The KL divergence for the mixture of Dirichlets.
            - kl_gaussian_loss: The KL divergence for the Gaussian distributions.

    """
    batch_size, n_mixtures = mix_logits.shape
    mix_probs = func.softmax(mix_logits, dim=-1)

    # Component-wise losses
    recon_losses_per_mix = []
    kl_gaussian_per_mix = []

    for m in range(n_mixtures):
        # MSE for mixture branch m
        mse_m = func.mse_loss(x_recon[:, m], x_true, reduction="mean")
        recon_losses_per_mix.append(mse_m)

        # Local Gaussian KL for mixture branch m using the extracted helper
        kl_gauss_m = _kl_divergence_gaussian(mu_q[:, m], logvar_q[:, m], mu_prior[:, m], logvar_prior[:, m])
        kl_gaussian_per_mix.append(kl_gauss_m)

    # Pack component losses into tensors: (batch, n_mixtures)
    recon_losses_tensor = torch.stack(recon_losses_per_mix, dim=-1)  # Broadcasting/scalar handling
    if recon_losses_tensor.dim() == 1:
        recon_losses_tensor = recon_losses_tensor.expand(batch_size, -1)

    kl_gaussian_tensor = torch.stack(kl_gaussian_per_mix, dim=-1)

    # Marginalize local tier tasks (expectation over q(c|x))
    marginalized_recon = (mix_probs * recon_losses_tensor).sum(dim=-1).mean()

    kl_gaussian_samples = (mix_probs * kl_gaussian_tensor).sum(dim=-1)
    kl_gaussian_samples = torch.clamp(kl_gaussian_samples, min=free_bits_gaussian)
    marginalized_kl_gaussian = kl_gaussian_samples.mean()

    # Mixture of Dirichlets analytical KL
    # Categorical component vs Uniform Prior
    kl_categorical = (
        mix_probs
        * (
            torch.log(mix_probs + 1e-6)
            + torch.log(torch.tensor(n_mixtures, device=mix_logits.device, dtype=mix_logits.dtype))
        )
    ).sum(dim=-1)

    # Evaluate Dirichlet-to-Dir(1) segments natively
    _, _, n_comp = alpha.shape
    alpha_flat = alpha.view(-1, n_comp)
    kl_dir_flat = _kl_divergence_dirichlet(alpha_flat).view(batch_size, n_mixtures)

    kl_mdm_samples = kl_categorical + (mix_probs * kl_dir_flat).sum(dim=-1)
    kl_mdm_samples = torch.clamp(kl_mdm_samples, min=free_bits_dirichlet)
    kl_mdm_loss = kl_mdm_samples.mean()

    # Total Expected Loss
    total_loss = (
        marginalized_recon + (kl_weight_dirichlet * kl_mdm_loss) + (kl_weight_gaussian * marginalized_kl_gaussian)
    )

    return total_loss, marginalized_recon, kl_mdm_loss, marginalized_kl_gaussian


def _kl_divergence_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
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


def _kl_divergence_gaussian(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
) -> torch.Tensor:
    """Compute the analytical KL divergence between two Gaussian profiles per antenna.

    KL(q(z_a | x) || p(z_a | s)) where q = N(mu_q, var_q) and p = N(mu_prior, var_prior)

    Arguments:
        mu_q (torch.Tensor): Posterior mean tensor of shape (batch_size, n_antennas, n_gaussians)
        logvar_q (torch.Tensor): Posterior log-variance tensor of shape (batch_size, n_antennas, n_gaussians)
        mu_prior (torch.Tensor): Data-dependent prior mean tensor of shape (batch_size, n_antennas, n_gaussians)
        logvar_prior (torch.Tensor): Data-dependent prior log-variance tensor
            of shape (batch_size, n_antennas, n_gaussians)

    Returns:
        kl_loss (torch.Tensor): KL divergence per sample averaged over antennas of shape (batch_size,)

    """
    var_q = torch.exp(logvar_q)
    var_prior = torch.exp(logvar_prior)

    kl_elemental = 0.5 * (logvar_prior - logvar_q + (var_q + (mu_q - mu_prior).pow(2)) / (var_prior + 1e-6) - 1.0)
    # Sum over the Gaussian latent components, average across antennas
    return kl_elemental.sum(dim=-1).mean(dim=1)
