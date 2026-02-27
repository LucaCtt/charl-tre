import torch
import torch.nn.functional as func
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

    @torch.no_grad()
    def get_shapes(self) -> tuple[tuple, int]:
        """Return the latent feature map shape and its flattened size.

        Returns:
            latent_feat_shape: The shape of the feature map after convolution (Channels, H, W).
            flat_dim: The total number of features when the feature map is flattened.

        """
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
        return self.__deconv(z)


class SingleAntennaVAE(nn.Module):
    """VAE architecture that encodes a single antenna's CSI data."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        latent_dim: int,
    ) -> None:
        """Initialize the SingleAntennaVAE with an encoder and decoder for single-antenna CSI data.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            latent_dim: The dimensionality of the latent space.

        """
        super().__init__()

        self.__encoder = AntennaEncoder(window_size, n_subcarriers, latent_dim)
        latent_feat_shape, flat_dim = self.__encoder.get_shapes()
        self.__decoder = AntennaDecoder(latent_feat_shape, flat_dim, latent_dim)

    def __reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the Gaussian distribution defined by mu and logvar."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input and sample a latent vector using the reparameterization trick."""
        mu, logvar = self.__encoder(x)
        return self.__reparameterize(mu, logvar), mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector to reconstruct the input."""
        return self.__decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input, sample a global categorical latent variable, and decode to reconstruct the input.

        Arguments:
            x: Input tensor of shape (batch_size, 1, window_size, n_subcarriers).

        Returns:
            recon: Tensor of shape (batch_size, 1, window_size, n_subcarriers) representing the reconstructed input.
            z: Tensor of shape (batch_size, 1, antenna_latent_dim) representing the sampled latent vector.
            mu: Tensor of shape (batch_size, 1, antenna_latent_dim) representing the mean of the latent vector.
            logvar: Tensor of shape (batch_size, 1, antenna_latent_dim) representing the log-variance of the latent.

        """
        z, mu, logvar = self.encode(x)
        recon = self.decode(z)

        return recon, z, mu, logvar
