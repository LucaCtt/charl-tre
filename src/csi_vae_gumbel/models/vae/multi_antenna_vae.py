from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as func
#from sklearn.manifold import TSNE
from torch import nn


class AntennaEncoder(nn.Module):
    """Encode a single-antenna CSI window into mean and log-variance vectors."""

    def __init__(self, window_size: int, n_subcarriers: int, antenna_latent_dim: int) -> None:
        """Initialize the AntennaEncoder with convolutional layers and linear heads.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            antenna_latent_dim: The dimensionality of the latent space for each antenna

        """
        super().__init__()
        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers

        # Convolutional feature extractor over time-frequency input
        self.__conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 8), stride=(5, 8)),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(5, 8), stride=(5, 8)),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 4), stride=(1, 1)),
            nn.GELU(),
            nn.Flatten(),
        )
        # Infer flattened feature dimension for linear heads
        _, flat_dim = self.get_shapes()

        # Linear heads for Gaussian parameters
        self.__mu = nn.Linear(flat_dim, antenna_latent_dim)
        self.__logvar = nn.Linear(flat_dim, antenna_latent_dim)

    def get_shapes(self) -> tuple[tuple, int]:
        """Return the latent feature map shape and its flattened size.

        Returns:
            latent_feat_shape: The shape of the feature map after convolution (Channels, H, W).
            flat_dim: The total number of features when the feature map is flattened.

        """
        with torch.no_grad():
            # Mock input to trace conv output shape
            x = torch.zeros(1, 1, self.__window_size, self.__n_subcarriers)
            x = self.__conv[:-1](x)

            latent_feat_shape = x.shape[1:]
            flat_dim = int(torch.prod(torch.tensor(latent_feat_shape)).item())

            return latent_feat_shape, flat_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log-variance for a single-antenna input.

        Arguments:
            x: Input tensor of shape (batch_size, window_size, n_subcarriers) for one antenna.

        Returns:
            mu: Tensor of shape (batch_size, antenna_latent_dim) representing the mean of the latent distribution.
            logvar: Tensor of shape (batch_size, antenna_latent_dim)

        """
        # Add channel dimension: (batch_size, 1, window_size, n_subcarriers)
        x = x.unsqueeze(1)
        z = self.__conv(x)
        return self.__mu(z), self.__logvar(z)


class AntennaDecoder(nn.Module):
    """Decode a latent vector back into a CSI window for a single antenna."""

    def __init__(
        self,
        latent_feat_shape: tuple,
        flat_dim: int,
        antenna_latent_dim: int,
    ) -> None:
        """Initialize the AntennaDecoder with linear and deconvolutional layers.

        Arguments:
            latent_feat_shape: The shape of the feature map before flattening in the encoder (Channels, H, W).
            flat_dim: The total number of features when the feature map is flattened.
            antenna_latent_dim: The dimensionality of the latent space for each antenna.

        """
        super().__init__()

        self.__latent_feat_shape = latent_feat_shape

        # Decoder group
        self.__fc = nn.Linear(antenna_latent_dim, flat_dim)
        self.__deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(3, 4), stride=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 8), stride=(5, 8)),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(5, 8), stride=(5, 8)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector into a CSI window.

        Arguments:
            z: Input tensor of shape (batch_size, antenna_latent_dim) representing the latent vector for one antenna.

        Returns:
            recon: Tensor of shape (batch_size, window_size, n_subcarriers)
                   representing the reconstructed CSI window for one antenna.

        """
        z = func.gelu(self.__fc(z))
        z = z.view(-1, *self.__latent_feat_shape)
        z = self.__deconv(z)
        return z.squeeze(1)  # Remove channel dimension: (Batch, window_size, n_subcarriers)


class MultiAntennaVAE(nn.Module):
    """VAE architecture that encodes multiple antennas separately and samples a global categorical latent variable."""

    def __init__(
        self,
        n_antennas: int,
        window_size: int,
        n_subcarriers: int,
        n_categories: int,
        latent_dim: int,
        antenna_latent_dim: int = 3,
    ) -> None:
        """Initialize the MultiAntennaVAE with separate encoders/decoders for each antenna and a global sampling layer.

        Arguments:
            n_antennas: The total number of antennas in the input data.
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_categories: The number of categories for the Gumbel-Softmax distribution (global latent variable).
            latent_dim: The dimensionality of the global latent space after sampling.
            antenna_latent_dim: The dimensionality of the latent space for each individual antenna.

        """
        super().__init__()

        self.__n_antennas = n_antennas
        self.__n_categories = n_categories
        self.__latent_dim = latent_dim

        self.__antenna_encoders = nn.ModuleList(
            [AntennaEncoder(window_size, n_subcarriers, antenna_latent_dim) for _ in range(n_antennas)],
        )
        self.__encoder_fc = nn.Linear(n_antennas * antenna_latent_dim, n_categories * latent_dim)

        latent_feat_shape, flat_dim = cast("AntennaEncoder", self.__antenna_encoders[0]).get_shapes()

        self.__decoder_fc = nn.Linear(n_categories * latent_dim, n_antennas * antenna_latent_dim)
        self.__antenna_decoders = nn.ModuleList(
            [AntennaDecoder(latent_feat_shape, flat_dim, antenna_latent_dim) for _ in range(n_antennas)],
        )


    def __reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the Gaussian distribution defined by mu and logvar."""
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input, sample a global categorical latent variable, and decode to reconstruct the input.

        Arguments:
            x: Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers).
            tau: The temperature parameter for the Gumbel-Softmax sampling.

        Returns:
            recons: Tensor of shape (batch_size, n_antennas, window_size, n_subcarriers)
                    representing the reconstructed CSI windows for all antennas.
            mus: Tensor of shape (batch_size, n_antennas, antenna_latent_dim)
                 containing the mean vectors from each antenna encoder.
            logvars: Tensor of shape (batch_size, n_antennas, antenna_latent_dim)
                     containing the log-variance vectors from each antenna encoder.
            z_hard: Tensor of shape (batch_size, n_categories, latent_dim)
                    representing the one-hot encoded global latent variable sampled via Gumbel-Softmax.
            logits: Tensor of shape (batch_size, n_categories, latent_dim)
                    representing the pre-softmax logits for the global latent variable.

        """
        batch_size = x.size(0)
        antenna_mus, antenna_logvars, antenna_latents = [], [], []

        # Encode each antenna
        for i in range(self.__n_antennas):
            mu, logvar = self.__antenna_encoders[i](x[:, i])
            z = self.__reparameterize(mu, logvar)
            antenna_mus.append(mu)
            antenna_logvars.append(logvar)
            antenna_latents.append(z)

        # Stack to (batch_size, n_antennas, antenna_latent_dim)
        mus = torch.stack(antenna_mus, dim=1)
        logvars = torch.stack(antenna_logvars, dim=1)
        latents = torch.stack(antenna_latents, dim=1)

        # Flatten and compute logits for global categorical sampling
        flattened_latents = latents.view(batch_size, -1)
        logits = self.__encoder_fc(flattened_latents)

        # Global categorical sampling
        logits = logits.view(batch_size, self.__latent_dim, self.__n_categories)
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        # Decode using the sampled global latent variable and unflatten
        decoded_latents = self.__decoder_fc(z_hard.view(batch_size, -1))
        decoded_latents = decoded_latents.view(batch_size, self.__n_antennas, -1)

        # Decode each antenna
        antenna_recons = []
        for i in range(self.__n_antennas):
            recon_i = self.__antenna_decoders[i](decoded_latents[:, i])
            antenna_recons.append(recon_i)

        # Stack reconstructions to get (batch_size, n_antennas, window_size, n_subcarriers)
        recon = torch.stack(antenna_recons, dim=1)

        return recon, mus, logvars, z_hard, logits
