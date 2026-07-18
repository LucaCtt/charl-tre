import torch
from torch import nn

from charl_tre.models import dirichlet, gaussian, util


class Fusion(nn.Module):
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
            flat_dim * n_antennas,
            total_mdm_outputs,
            n_fusion_layers,
            fusion_dropout,
        )

        # Top-down prior generators
        self._top_down_priors = nn.ModuleList([
            nn.Linear(
                n_dirichlet_components,
                n_gaussians_per_antenna * 2,
            )
            for _ in range(n_antennas)
        ])

    def _precision_weighted_merge(
        self,
        mu_bu: torch.Tensor,
        logvar_bu: torch.Tensor,
        mu_prior: torch.Tensor,
        logvar_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        var_bu = torch.exp(logvar_bu)
        var_prior = torch.exp(logvar_prior)

        precision_bu = 1.0 / (var_bu + self._eps)
        precision_prior = 1.0 / (var_prior + self._eps)

        var_q = 1.0 / (precision_bu + precision_prior + self._eps)
        mu_q = (mu_bu * precision_bu + mu_prior * precision_prior) * var_q
        logvar_q = torch.log(var_q + self._eps)

        return mu_q, logvar_q

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with Mixture of Dirichlets allocation and Ladder merging.

        Arguments:
            x (torch.Tensor): Input antenna signals of shape (batch, n_antennas, window_size, n_subcarriers)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - recon (torch.Tensor): Reconstructed antenna signals
                - mix_logits (torch.Tensor): Categorical assignment logits of shape (batch, n_mixtures)
                - alpha (torch.Tensor): Distributed concentrations of shape (batch, n_mixtures, n_components)
                - mu_q (torch.Tensor): Posteriors cross-evaluated over antennas
                - logvar_q (torch.Tensor): Log-variances of posteriors
                - mu_prior (torch.Tensor): Conditional data-dependent priors
                - logvar_prior (torch.Tensor): Log-variances of priors

        """
        batch_size = x.shape[0]
        h_list, mu_bu_list, logvar_bu_list = [], [], []

        # Bottom-up encoding of each antenna's signal
        for i, gaussian_vae in enumerate(self._gaussians):
            h_a, mu_bu, logvar_bu = gaussian_vae.encode(x[:, i])  # pyright: ignore[reportCallIssue]
            h_list.append(h_a)
            mu_bu_list.append(mu_bu)
            logvar_bu_list.append(logvar_bu)

        h_concat = torch.cat(h_list, dim=1)

        # Map compressed features to Mixture of Dirichlets parameters
        mdm_params = self._dirichlet_in(h_concat)

        # Separate mixture categories from concentration updates
        mix_logits = mdm_params[:, : self._n_mixtures]
        alpha_raw = mdm_params[:, self._n_mixtures :]
        alpha = (
            nn.functional.softplus(alpha_raw.view(batch_size, self._n_mixtures, self._n_dirichlet_components))
            + self._eps
        )

        # Sample Dirichlet space for ALL mixtures simultaneously: (batch, n_mixtures, n_components)
        s_components = dirichlet.Autoencoder.reparameterize(alpha)

        # Flatten the Batch and Mixture axes into a unified mega-batch dimension: (Batch * N_Mixtures, N_Components)
        s_flat = s_components.view(batch_size * self._n_mixtures, self._n_dirichlet_components)

        mu_q_all_mix, logvar_q_all_mix = [], []
        mu_p_all_mix, logvar_p_all_mix = [], []
        recons_all_mix = []

        # Loop over antennas remains sequential, but all mixture branches are processed simultaneously
        for i, gaussian_vae in enumerate(self._gaussians):
            # 1. Forward pass through top-down MLPs for ALL mixtures at once
            prior_params_flat = self._top_down_priors[i](s_flat)
            mu_prior_flat, logvar_prior_flat = torch.chunk(prior_params_flat, 2, dim=-1)
            logvar_prior_flat = torch.clamp(logvar_prior_flat, min=-5, max=5)

            # 2. Replicate localized antenna bottom-ups to align with mega-batch shape
            # Original: (Batch, Latents) -> Intermediary: (Batch, 1, Latents)
            #   -> Broadcasted: (Batch, N_Mixtures, Latents)
            # Final Flat: (Batch * N_Mixtures, Latents)
            mu_bu_flat = mu_bu_list[i].repeat_interleave(self._n_mixtures, dim=0)
            logvar_bu_flat = logvar_bu_list[i].repeat_interleave(self._n_mixtures, dim=0)

            # 3. Mass Precision-Weighted Ladder Merge
            mu_q_flat, logvar_q_flat = self._precision_weighted_merge(
                mu_bu_flat,
                logvar_bu_flat,
                mu_prior_flat,
                logvar_prior_flat,
            )

            # 4. Standard Reparameterization Sampling
            z_a_flat = gaussian.Autoencoder.reparameterize(mu_q_flat, logvar_q_flat)

            # 5. Decode the global batch array
            recon_flat = gaussian_vae.decode(z_a_flat)  # pyright: ignore[reportCallIssue]

            # 6. Unflatten back to structured separate mixture axes
            recons_all_mix.append(recon_flat.view(batch_size, self._n_mixtures, *recon_flat.shape[1:]))
            mu_q_all_mix.append(mu_q_flat.view(batch_size, self._n_mixtures, -1))
            logvar_q_all_mix.append(logvar_q_flat.view(batch_size, self._n_mixtures, -1))
            mu_p_all_mix.append(mu_prior_flat.view(batch_size, self._n_mixtures, -1))
            logvar_p_all_mix.append(logvar_prior_flat.view(batch_size, self._n_mixtures, -1))

        # --- STEP 3: Permute back to the expected loss format ---
        # Current list layout: N_Antennas tensors of shape (Batch, N_Mixtures, ...)
        # Target format: (Batch, N_Mixtures, N_Antennas, ...)
        recon_tensor = torch.stack(recons_all_mix, dim=2)
        mu_q_tensor = torch.stack(mu_q_all_mix, dim=2)
        logvar_q_tensor = torch.stack(logvar_q_all_mix, dim=2)
        mu_p_tensor = torch.stack(mu_p_all_mix, dim=2)
        logvar_p_tensor = torch.stack(logvar_p_all_mix, dim=2)

        return (
            recon_tensor,
            mix_logits,
            alpha,
            mu_q_tensor,
            logvar_q_tensor,
            mu_p_tensor,
            logvar_p_tensor,
        )
