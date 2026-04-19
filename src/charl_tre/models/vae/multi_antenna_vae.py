from typing import cast

import torch
import torch.nn.functional as func
from torch import nn

from charl_tre.models.vae.gaussian.single_antenna_vae import SingleAntennaVAE


class MultiAntennaVAE(nn.Module):
    """VAE for multiple antennas represented with gaussian VAEs with a shared categorical latent space."""

    def __init__(
        self,
        antennas: nn.ModuleList,
        n_categories: int,
        latent_dim: int,
        antenna_latent_dim: int,
    ) -> None:
        """Initialize the MultiAntennaVAE."""
        super().__init__()

        self.__antennas = antennas
        self.__n_antennas = len(antennas)
        self.__n_categories = n_categories
        self.__latent_dim = latent_dim

        # Private endoders for each antenna
        self.__antenna_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(antenna_latent_dim, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Linear(256, 128),  # Intermediate feature size
                    nn.GELU(),
                )
                for _ in range(self.__n_antennas)
            ],
        )

        # 2. Global Fusion (Combined Features -> Categorical Logits)
        self.__global_fusion = nn.Sequential(
            nn.Linear(128 * self.__n_antennas, 512),
            nn.GELU(),
            nn.Linear(512, n_categories * latent_dim),
        )

        # 3. Private decoders for each antenna
        self.__antenna_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_categories * latent_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, antenna_latent_dim),  # MATCHES INPUT DIM
                )
                for _ in range(self.__n_antennas)
            ],
        )

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the multi-antenna VAE.

        Arguments:
            x: Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers)
            tau: Temperature for Gumbel-Softmax sampling

        Returns:
            recon: Reconstructed input of shape (batch_size, n_antennas, window_size, n_subcarriers)
            z_hard: Sampled categorical latent of shape (batch_size, latent_dim * n_categories)
            logits: Logits for the categorical distribution of shape (batch_size, latent_dim * n_categories)

        """
        batch_size = x.size(0)

        # --- ENCODE ---
        original_mus = []
        private_features = []
        for i, antenna in enumerate(self.__antennas):
            # Capture the original mu for Feature Matching
            antenna = cast("SingleAntennaVAE", antenna)
            _, mu, _ = antenna.encode(x[:, i].unsqueeze(1))
            original_mus.append(mu)

            # Project to the high-dim space for fusion
            private_features.append(self.__antenna_encoders[i](mu))

        # --- GLOBAL BOTTLENECK ---
        combined_features = torch.cat(private_features, dim=1)
        logits = self.__global_fusion(combined_features)
        logits = logits.view(batch_size, self.__latent_dim, self.__n_categories)
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        z_flat = z_hard.view(batch_size, -1)

        # --- DECODE ---
        antenna_recons = []
        latents_post = []
        for i, antenna in enumerate(self.__antennas):
            # Extract the antenna-specific latent from the global code
            ant_lat_recon = self.__antenna_decoders[i](z_flat)
            latents_post.append(ant_lat_recon)

            # Sub-VAE Decoder expects (B, antenna_latent_dim)
            recon_i = cast("SingleAntennaVAE", antenna).decode(ant_lat_recon).squeeze(1)
            antenna_recons.append(recon_i)

        recon = torch.stack(antenna_recons, dim=1)
        latents_post = torch.stack(latents_post, dim=1)

        return recon, z_hard, logits
