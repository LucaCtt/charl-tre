import torch
from torch import nn


class _Encoder(nn.Module):
    """Encode a single-antenna CSI window into mean and log-variance vectors."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: list[tuple[int, int]],
    ) -> None:
        """Initialize the AntennaEncoder with convolutional layers and linear heads.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The dimensionality of the latent space.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()
        self._window_size = window_size
        self._n_subcarriers = n_subcarriers

        layers: list[nn.Module] = []
        for kh, sh in conv_layers:
            layers.append(nn.Conv2d(n_subcarriers, n_subcarriers, kernel_size=(kh, 1), stride=(sh, 1)))
            layers.append(nn.BatchNorm2d(n_subcarriers))
            layers.append(nn.GELU())

        layers.append(nn.Flatten())
        self._conv = nn.Sequential(*layers)

        # Infer flattened feature dimension for linear heads
        _, flat_dim = self.get_shapes()

        # Linear heads for Gaussian parameters
        self._mu = nn.Linear(flat_dim, n_gaussians)
        self._logvar = nn.Linear(flat_dim, n_gaussians)

    @torch.no_grad()
    def get_shapes(self) -> tuple[tuple, int]:
        """Return the latent feature map shape and its flattened size.

        Returns:
            latent_feat_shape: The shape of the feature map after convolution (Channels, H, W).
            flat_dim: The total number of features when the feature map is flattened.

        """
        x = torch.zeros(1, self._n_subcarriers, self._window_size, 1, device=next(self.parameters()).device)
        was_training = self._conv.training
        self._conv.eval()
        x = self._conv[:-1](x)
        self._conv.train(was_training)
        return x.shape[1:], int(x.numel() // x.shape[0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and log-variance for a single-antenna input.

        Arguments:
            x: Input tensor of shape (batch_size, window_size, n_subcarriers) for one antenna.

        Returns:
            mu: Tensor of shape (batch_size, antenna_n_gaussians) representing the mean of the latent distribution.
            logvar: Tensor of shape (batch_size, antenna_n_gaussians)

        """
        x = x.permute(0, 2, 1).unsqueeze(-1).contiguous()  # (batch_size, n_subcarriers, window_size, 1)
        z = self._conv(x)
        return self._mu(z), torch.clamp(self._logvar(z), min=-10, max=10)


class _Decoder(nn.Module):
    """Decode a latent vector back into a CSI window for a single antenna."""

    def __init__(
        self,
        latent_feat_shape: tuple,
        flat_dim: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: list[tuple[int, int]],
    ) -> None:
        """Initialize the AntennaDecoder with linear and deconvolutional layers.

        Arguments:
            latent_feat_shape: The shape of the feature map before flattening in the encoder.
            flat_dim: The total number of features when the feature map is flattened.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The number of gaussians to decode from.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride)

        """
        super().__init__()
        self._latent_feat_shape = latent_feat_shape

        self._fc = nn.Sequential(nn.Linear(n_gaussians, flat_dim), nn.GELU())

        deconv_layers: list[nn.Module] = []
        reversed_specs = list(reversed(conv_layers))

        for i, (kh, sh) in enumerate(reversed_specs):
            deconv_layers.append(
                nn.ConvTranspose2d(n_subcarriers, n_subcarriers, kernel_size=(kh, 1), stride=(sh, 1)),
            )
            if i < len(reversed_specs) - 1:
                deconv_layers.append(nn.BatchNorm2d(n_subcarriers))
                deconv_layers.append(nn.GELU())

        self._deconv = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector into a CSI window.

        Arguments:
            z: Input tensor of shape (batch_size, n_gaussians*2) representing the latent vector for one antenna.

        Returns:
            recon: Tensor of shape (batch_size, window_size, n_subcarriers)
                   representing the reconstructed CSI window for one antenna.

        """
        z = self._fc(z)  # (batch_size, flat_dim)
        z = z.view(z.size(0), *self._latent_feat_shape).contiguous()  # (batch_size, n_subcarriers, window_size', 1)
        z = self._deconv(z)  # (batch_size, n_subcarriers, window_size, 1)
        return z.squeeze(-1).permute(0, 2, 1)  # (batch_size, window_size, n_subcarriers)


class Autoencoder(nn.Module):
    """Gaussian autoencoder for CSI data, consisting of an encoder and decoder."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: list[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize the Gaussian autoencoder with an encoder and decoder for single-antenna CSI data.

        Arguments:
            window_size: The size of the time window for CSI input.
            n_subcarriers: The number of subcarriers in the CSI input.
            n_gaussians: The number of gaussians to encode/decode in the latent space.
            conv_layers: A list of tuples specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()

        if conv_layers is None:
            conv_layers = [(5, 5), (5, 5), (3, 3)]

        self._encoder = _Encoder(window_size, n_subcarriers, n_gaussians, conv_layers)
        latent_feat_shape, flat_dim = self._encoder.get_shapes()
        self._decoder = _Decoder(latent_feat_shape, flat_dim, n_subcarriers, n_gaussians, conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(2, window_size, n_subcarriers)
            recon, _, _ = self.forward(dummy)
            if recon.shape != dummy.shape:
                msg = f"Decoder output shape {recon.shape} does not match input shape {dummy.shape}"
                raise ValueError(msg)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the Gaussian distribution defined by mu and logvar."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input CSI window into mean and log-variance vectors."""
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector to reconstruct the input."""
        return self._decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input, sample a latent variable, and decode to reconstruct the input.

        Arguments:
            x: Input tensor of shape (batch_size, window_size, n_subcarriers).

        Returns:
            recon: Tensor of shape (batch_size, window_size, n_subcarriers) representing the reconstructed input.
            mu: Tensor of shape (batch_size, n_gaussians) representing the mean of the latent vector.
            logvar: Tensor of shape (batch_size, n_gaussians) representing the log-variance of the latent.

        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar
