from typing import cast

import torch
import torch.nn.functional as func
from torch import nn


class AntennaEncoder(nn.Module):
    def __init__(self, window_size: int, n_subcarriers: int, antenna_latent_dim: int) -> None:
        super().__init__()
        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers

        self.__conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 8), stride=(2, 4), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=(2, 3), padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        _, flat_dim = self.get_shapes()

        self.__mu = nn.Linear(flat_dim, antenna_latent_dim)
        self.__logvar = nn.Linear(flat_dim, antenna_latent_dim)

    def get_shapes(self) -> tuple[tuple, int]:
        """Mock pass to find flattened size and required output_paddings."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.__window_size, self.__n_subcarriers)

            # Trace Layer 1
            l1 = self.__conv[0](x)
            # Trace Layer 2
            l2 = self.__conv[2](l1)
            # Trace Layer 3
            l3 = self.__conv[4](l2)
            # Trace Layer 4
            l4 = self.__conv[6](l3)

            latent_feat_shape = l4.shape[1:]
            flat_dim = int(torch.prod(torch.tensor(latent_feat_shape)).item())

            return latent_feat_shape, flat_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # Add channel dimension: (Batch, 1, window_size, n_subcarriers)
        z = self.__conv(x)
        return self.__mu(z), self.__logvar(z)


class AntennaDecoder(nn.Module):
    def __init__(
        self,
        latent_feat_shape: tuple,
        flat_dim: int,
        antenna_latent_dim: int,
    ) -> None:
        super().__init__()

        self.__latent_feat_shape = latent_feat_shape

        # Decoder group
        self.__fc = nn.Linear(antenna_latent_dim, flat_dim)
        self.__deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=(2, 3), padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2, 2), padding=1, output_padding=(0, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 5), stride=(2, 4), padding=(1, 2), output_padding=(1, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(3, 8), stride=(2, 4), padding=(1, 2), output_padding=(0, 0)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Map back to 4D tensor: [Batch, Channels, H, W]
        z = func.relu(self.__fc(z))
        z = z.view(-1, *self.__latent_feat_shape)
        z = self.__deconv(z)
        return z.squeeze(1)  # Remove channel dimension: (Batch, window_size, n_subcarriers)


class MultiAntennaVAE(nn.Module):
    def __init__(
        self,
        n_antennas: int,
        window_size: int,
        n_subcarriers: int,
        n_categories: int,
        latent_dim: int,
        antenna_latent_dim: int = 2,
    ) -> None:
        super().__init__()

        self.__n_antennas = n_antennas

        self.__antenna_encoders = nn.ModuleList(
            [AntennaEncoder(window_size, n_subcarriers, antenna_latent_dim) for _ in range(n_antennas)],
        )
        self.__encoder_fc = nn.Linear(n_antennas * antenna_latent_dim, n_categories * latent_dim)

        self.__decoder_fc = nn.Linear(n_categories * latent_dim, n_antennas * antenna_latent_dim)
        latent_feat_shape, flat_dim = cast(
            "AntennaEncoder",
            self.__antenna_encoders[0],
        ).get_shapes()
        self.__antenna_decoders = nn.ModuleList(
            [AntennaDecoder(latent_feat_shape, flat_dim, antenna_latent_dim) for _ in range(n_antennas)],
        )

    def __reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        # x shape: (batch, n_antennas, input_dim)
        batch_size = x.size(0)
        all_mus, all_logvars, all_zs = [], [], []

        # Step 1: Encode each antenna individually
        for i in range(self.__n_antennas):
            mu, logvar = self.__antenna_encoders[i](x[:, i])
            z = self.__reparameterize(mu, logvar)
            all_mus.append(mu)
            all_logvars.append(logvar)
            all_zs.append(z)

        # Stack to (Batch, n_antennas, antenna_latent_dim)
        mus = torch.stack(all_mus, dim=1)
        logvars = torch.stack(all_logvars, dim=1)
        zs = torch.stack(all_zs, dim=1)

        # Step 2: Global Categorical Sampling
        # Flatten latents: (Batch, n_antennas * antenna_latent_dim)
        z_flattened = zs.view(batch_size, -1)
        logits = self.__encoder_fc(z_flattened)

        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True)

        recon = self.__decoder_fc(z_hard)
        recon = recon.view(batch_size, self.__n_antennas, -1)

        # Reconstruction (Using the continuous latents zs)
        all_recons = []
        for i in range(self.__n_antennas):
            recon_i = self.__antenna_decoders[i](recon[:, i])
            all_recons.append(recon_i)

        recons = torch.stack(all_recons, dim=1)  # (Batch, n_antennas, window_size, n_subcarriers)

        return recons, mus, logvars, z_hard, logits
