import torch
from torch import nn

from charl_tre.models.common import util
from charl_tre.models.early import gaussian


class Fusion(nn.Module):
    """Early fusion: Gaussian VAEs per antenna + Dirichlet for shared latent."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_dirichlet_components: int,
        dirichlet_conv_layers: list[tuple[int, int]],
        n_fusion_layers: int,
        fusion_dropout: float,
        n_antennas: int,
        n_gaussians_per_antenna: int = 2,
    ) -> None:
        """Initialize early fusion VAE.

        Arguments:
            window_size (int): CSI window size.
            n_subcarriers (int): Number of subcarriers.
            n_dirichlet_components (int): Dirichlet latent dimension.
            dirichlet_conv_layers (list[tuple[int, int]]): List of (kernel_size, stride) for Dirichlet conv layers.
            n_fusion_layers (int): Number of FC layers before and after Dirichlet.
            fusion_dropout (float): Dropout in fusion layers.
            n_antennas (int): Number of antennas.
            n_gaussians_per_antenna (int): Number of Gaussian VAEs per antenna.

        """
        super().__init__()

        self._n_antennas = n_antennas
        self._n_dirichlet_components = n_dirichlet_components
        self._n_gaussians_per_antenna = n_gaussians_per_antenna
        self._eps = 1e-4

        self._gaussians = nn.ModuleList()
        for _ in range(n_antennas):
            self._gaussians.append(gaussian.Autoencoder(window_size, n_subcarriers, n_gaussians_per_antenna))

        concat_gaussian_dim = n_gaussians_per_antenna * 2 * n_antennas

        self._dirichlet_in = util.build_fc(
            concat_gaussian_dim,
            n_dirichlet_components,
            n_fusion_layers,
            fusion_dropout,
        )
        self._dirichlet_out = util.build_fc(
            n_dirichlet_components,
            concat_gaussian_dim,
            n_fusion_layers,
            fusion_dropout,
        )

    def _reparameterize(self, alpha: torch.Tensor) -> torch.Tensor:
        """Sample latent Dirichlet vector via Gamma reparameterization."""
        gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=torch.ones_like(alpha))
        gamma_samples = gamma_dist.rsample()
        return gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode to Dirichlet alpha.

        Arguments:
            x (torch.Tensor): Shape (batch_size, n_antennas, window_size, n_subcarriers).

        Returns:
            alpha (torch.Tensor): Dirichlet concentration parameters (batch_size, n_components).
            gaussian_z (torch.Tensor): Concatenated Gaussian latents (batch_size, gaussian_latent_dim * n_antennas).

        """
        gaussian_outs = []
        for i, vae in enumerate(self._gaussians):
            if not isinstance(vae, gaussian.Autoencoder):
                msg = f"Expected Autoencoder, got {type(vae)}"
                raise TypeError(msg)

            mu, logvar = vae.encode(x[:, i])
            gaussian_outs.append(torch.cat([mu, logvar], dim=1))

        gaussian_z = torch.cat(gaussian_outs, dim=1)
        linear_z = self._dirichlet_in(gaussian_z)
        alpha = nn.functional.softplus(linear_z) + self._eps

        return alpha, gaussian_z

    def decode(self, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from Dirichlet alpha to antenna reconstructions.

        Arguments:
            alpha (torch.Tensor): Dirichlet parameters (batch_size, n_components).

        Returns:
            recon (torch.Tensor): Reconstructed antenna signals (batch_size, n_antennas, window_size, n_subcarriers).
            gaussian_z (torch.Tensor): Concatenated Gaussian latents (batch_size, gaussian_latent_dim * n_antennas).

        """
        dirichlet_z = self._reparameterize(alpha)
        gaussian_z = self._dirichlet_out(dirichlet_z)

        per_antenna_dim = self._n_gaussians_per_antenna * 2
        gaussian_outs = torch.split(gaussian_z, per_antenna_dim, dim=1)

        recons = []
        for i, vae in enumerate(self._gaussians):
            if not isinstance(vae, gaussian.Autoencoder):
                msg = f"Expected Autoencoder, got {type(self._gaussians[i])}"
                raise TypeError(msg)

            gaussian_out = gaussian_outs[i]
            x = vae.reparameterize(
                gaussian_out[:, : self._n_gaussians_per_antenna],
                gaussian_out[:, self._n_gaussians_per_antenna :],
            )
            x = vae.decode(x)
            recons.append(x)

        return torch.stack(recons, dim=1), gaussian_z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Arguments:
            x (torch.Tensor): Shape (batch_size, n_antennas, window_size, n_subcarriers).

        Returns:
            recon (torch.Tensor): Reconstructed antenna signals.
            alpha (torch.Tensor): Dirichlet parameters.

        """
        alpha, _ = self.encode(x)
        recon, _ = self.decode(alpha)

        return recon, alpha
