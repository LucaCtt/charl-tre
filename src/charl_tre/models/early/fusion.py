import torch
from torch import nn

from charl_tre.models.common import util
from charl_tre.models.early import gaussian


class HierarchicalFusion(nn.Module):
    """Hierarchical Ladder VAE using a global Mixture of Dirichlets (MDM) posterior.

    Permutation-invariant pooling across antennas routes through discrete components
    to condition downstream localized Gaussian priors.
    """

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_dirichlet_components: int,
        n_mixtures: int,
        n_fusion_layers: int,
        fusion_dropout: float,
        n_antennas: int,
        n_gaussians_per_antenna: int = 2,
        eps: float = 1e-4,
    ) -> None:
        """Initialize the HierarchicalFusion model.

        Arguments:
            window_size (int): Size of the input window for each antenna signal.
            n_subcarriers (int): Number of subcarriers in the input signal.
            n_dirichlet_components (int): Number of components in the Dirichlet distribution.
            n_mixtures (int): Number of mixture components in the MDM.
            n_fusion_layers (int): Number of fully connected layers in the fusion network.
            fusion_dropout (float): Dropout rate for the fusion network.
            n_antennas (int): Number of antennas in the input signal.
            n_gaussians_per_antenna (int): Number of Gaussian components per antenna.
            eps (float): Small constant for numerical stability.

        """
        super().__init__()

        self._n_antennas = n_antennas
        self._n_dirichlet_components = n_dirichlet_components
        self._n_mixtures = n_mixtures
        self._n_gaussians_per_antenna = n_gaussians_per_antenna
        self._eps = eps

        self._gaussians = nn.ModuleList(
            [gaussian.Autoencoder(window_size, n_subcarriers, n_gaussians_per_antenna) for _ in range(n_antennas)],
        )

        _, flat_dim = self._gaussians[0].get_shapes()  # pyright: ignore[reportCallIssue]

        # Outputs component probabilities (logits) + concentration parameters for each mixture
        total_mdm_outputs = n_mixtures + (n_mixtures * n_dirichlet_components)
        self._dirichlet_in = util.build_fc(
            flat_dim,
            total_mdm_outputs,
            n_fusion_layers,
            fusion_dropout,
        )

        # Top-down prior generators
        self._top_down_priors = nn.ModuleList([
            util.build_fc(
                n_dirichlet_components,
                n_gaussians_per_antenna * 2,
                n_fusion_layers,
                fusion_dropout,
            )
            for _ in range(n_antennas)
        ])

    def _reparameterize_dirichlet(self, alpha: torch.Tensor) -> torch.Tensor:
        gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=torch.ones_like(alpha))
        gamma_samples = gamma_dist.rsample()
        return gamma_samples / (gamma_samples.sum(dim=-1, keepdim=True) + self._eps)

    def _precision_weighted_merge(
        self,
        mu_bu: torch.Tensor,
        logvar_bu: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var_bu = torch.exp(logvar_bu)
        var_prior = torch.exp(logvar_prior)

        prec_bu = 1.0 / (var_bu + self._eps)
        prec_prior = 1.0 / (var_prior + self._eps)

        var_q = 1.0 / (prec_bu + prec_prior + self._eps)
        mu_q = (mu_bu * prec_bu + mu_prior * prec_prior) * var_q
        logvar_q = torch.log(var_q + self._eps)

        return mu_q, logvar_q

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with Mixture of Dirichlets allocation and Ladder merging.

        Arguments:
            x: Input antenna signals of shape (batch, n_antennas, window_size, n_subcarriers)

        Returns:
            recon: Reconstructed antenna signals
            mix_logits: Categorical assignment logits of shape (batch, n_mixtures)
            alpha: Distributed concentrations of shape (batch, n_mixtures, n_components)
            mu_q, logvar_q: Posteriors cross-evaluated over antennas
            mu_prior, logvar_prior: Conditional data-dependent priors

        """
        batch_size = x.shape[0]
        h_list, mu_bu_list, logvar_bu_list = [], [], []

        # Bottom-up encoding of each antenna's signal
        for i, gaussian_vae in enumerate(self._gaussians):
            h_a, mu_bu, logvar_bu = gaussian_vae.encode(x[:, i])  # pyright: ignore[reportCallIssue]
            h_list.append(h_a)
            mu_bu_list.append(mu_bu)
            logvar_bu_list.append(logvar_bu)

        h_pooled = torch.stack(h_list, dim=1).mean(dim=1)

        # Map compressed features to Mixture of Dirichlets parameters
        mdm_params = self._dirichlet_in(h_pooled)

        # Separate mixture categories from concentration updates
        mix_logits = mdm_params[:, : self._n_mixtures]
        alpha_raw = mdm_params[:, self._n_mixtures :]
        alpha = (
            nn.functional.softplus(alpha_raw.view(batch_size, self._n_mixtures, self._n_dirichlet_components))
            + self._eps
        )

        # Sample Dirichlet space for ALL mixtures simultaneously: (batch, n_mixtures, n_components)
        s_components = self._reparameterize_dirichlet(alpha)

        # Compute shared mixture representation by weighting each mixture's components
        recons_all_mix = []
        mu_q_all_mix, logvar_q_all_mix = [], []
        mu_p_all_mix, logvar_p_all_mix = [], []

        for m in range(self._n_mixtures):
            s_m = s_components[:, m, :]  # (batch, n_components)

            recons_m = []
            mu_q_list, logvar_q_list = [], []
            mu_prior_list, logvar_prior_list = [], []

            for i, gaussian_vae in enumerate(self._gaussians):
                prior_params = self._top_down_priors[i](s_m)
                mu_prior, logvar_prior = torch.chunk(prior_params, 2, dim=-1)
                logvar_prior = torch.clamp(logvar_prior, min=-10, max=10)

                # Merge with antenna-specific bottom-up parameters
                mu_q, logvar_q = self._precision_weighted_merge(
                    mu_bu_list[i],
                    logvar_bu_list[i],
                    mu_prior,
                    logvar_prior,
                )

                std_q = torch.exp(0.5 * logvar_q)
                z_a = mu_q + torch.randn_like(std_q) * std_q

                recons_m.append(gaussian_vae.decode(z_a))  # pyright: ignore[reportCallIssue]
                mu_q_list.append(mu_q)
                logvar_q_list.append(logvar_q)
                mu_prior_list.append(mu_prior)
                logvar_prior_list.append(logvar_prior)

            recons_all_mix.append(torch.stack(recons_m, dim=1))  # (batch, n_antennas, window, subcarriers)
            mu_q_all_mix.append(torch.stack(mu_q_list, dim=1))  # (batch, n_antennas, n_gaussians)
            logvar_q_all_mix.append(torch.stack(logvar_q_list, dim=1))
            mu_p_all_mix.append(torch.stack(mu_prior_list, dim=1))
            logvar_p_all_mix.append(torch.stack(logvar_prior_list, dim=1))

        return (
            torch.stack(recons_all_mix, dim=1),  # (batch, n_mixtures, n_antennas, window, subcarriers)
            mix_logits,  # (batch, n_mixtures)
            alpha,  # (batch, n_mixtures, n_components)
            torch.stack(mu_q_all_mix, dim=1),  # (batch, n_mixtures, n_antennas, n_gaussians)
            torch.stack(logvar_q_all_mix, dim=1),  # (batch, n_mixtures, n_antennas, n_gaussians)
            torch.stack(mu_p_all_mix, dim=1),  # (batch, n_mixtures, n_antennas, n_gaussians)
            torch.stack(logvar_p_all_mix, dim=1),  # (batch, n_mixtures, n_antennas, n_gaussians)
        )
