from typing import cast

import torch
import torch.nn.functional as func
from torch import nn

from csi_vae_gumbel.models.vae.single_antenna_vae import SingleAntennaVAE


class MultiAntennaVAE(nn.Module):
    def __init__(
        self,
        antennas: nn.ModuleList,
        n_categories: int,
        latent_dim: int,
        antenna_latent_dim: int,
    ) -> None:
        super().__init__()
        self.antennas = antennas
        self.__n_antennas = len(antennas)
        self.__n_categories = n_categories
        self.__latent_dim = latent_dim

        # 1. Private ENCODERS (Antenna -> High-dim Feature)
        self.enc_private = nn.ModuleList(
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
        self.global_fusion = nn.Sequential(
            nn.Linear(128 * self.__n_antennas, 512),
            nn.GELU(),
            nn.Linear(512, n_categories * latent_dim),
        )

        # 3. Private DECODERS (Global Z -> Original Antenna Latent)
        # This mirrors the encoder to return to exactly antenna_latent_dim
        self.dec_private = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_categories * latent_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, antenna_latent_dim),  # MATCHES INPUT DIM
                )
                for _ in range(self.__n_antennas)
            ]
        )

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        batch_size = x.size(0)

        # --- ENCODE ---
        original_mus = []
        private_features = []
        for i, antenna in enumerate(self.antennas):
            # Capture the original mu for Feature Matching
            _, mu, _ = antenna.encode(x[:, i].unsqueeze(1))
            original_mus.append(mu)

            # Project to the high-dim space for fusion
            private_features.append(self.enc_private[i](mu))

        # --- GLOBAL BOTTLENECK ---
        combined_features = torch.cat(private_features, dim=1)
        logits = self.global_fusion(combined_features)
        logits = logits.view(batch_size, self.__latent_dim, self.__n_categories)
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        z_flat = z_hard.view(batch_size, -1)

        # --- DECODE ---
        antenna_recons = []
        latents_post = []
        for i, antenna in enumerate(self.antennas):
            # Extract the antenna-specific latent from the global code
            ant_lat_recon = self.dec_private[i](z_flat)
            latents_post.append(ant_lat_recon)

            # Sub-VAE Decoder expects (B, antenna_latent_dim)
            recon_i = antenna.decode(ant_lat_recon).squeeze(1)
            antenna_recons.append(recon_i)

        recon = torch.stack(antenna_recons, dim=1)
        latents_pre = torch.stack(original_mus, dim=1)
        latents_post = torch.stack(latents_post, dim=1)

        return recon, z_hard, logits, latents_pre, latents_post
