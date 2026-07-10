import torch
from torch import nn


class _Encoder(nn.Module):
    """Encode a single-antenna CSI window into bottom-up features and distributions."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: list[tuple[int, int]],
    ) -> None:
        """Initialize the AntennaEncoder with convolutional layers and linear heads.

        Arguments:
            window_size (int): The size of the time window for CSI input.
            n_subcarriers (int): The number of subcarriers in the CSI input.
            n_gaussians (int): The dimensionality of the latent space.
            conv_layers (list[tuple[int, int]]): A list of tuples specifying
                the convolutional layers (kernel size and stride).

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

        _, flat_dim = self.get_shapes()

        # Bottom-up proposal heads
        self._mu_bu = nn.Linear(flat_dim, n_gaussians)
        self._logvar_bu = nn.Linear(flat_dim, n_gaussians)

        # Bottom-up feature head
        self._h_head = nn.Sequential(nn.Linear(flat_dim, flat_dim), nn.GELU())

    @torch.no_grad()
    def get_shapes(self) -> tuple[tuple, int]:
        """Get the shapes of the latent features and flattened dimension.

        Returns:
            latent_feat_shape (tuple): The shape of the latent features after convolution.
            flat_dim (int): The flattened dimension of the latent features.

        """
        x = torch.zeros(1, self._n_subcarriers, self._window_size, 1, device=next(self.parameters()).device)

        was_training = self._conv.training
        self._conv.eval()
        x = self._conv[:-1](x)
        self._conv.train(was_training)

        return x.shape[1:], int(x.numel() // x.shape[0])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input CSI window into bottom-up features and distributions.

        Args:
            x (torch.Tensor): Input CSI window of shape (batch_size, n_subcarriers, window_size).

        Returns:
            h_a (torch.Tensor): Bottom-up features.
            mu_bu (torch.Tensor): Mean of the bottom-up Gaussian distributions.
            logvar_bu (torch.Tensor): Log variance of the bottom-up Gaussian distributions.

        """
        x = x.permute(0, 2, 1).unsqueeze(-1).contiguous()
        z = self._conv(x)
        h_a = self._h_head(z)
        mu_bu = self._mu_bu(z)
        logvar_bu = torch.clamp(self._logvar_bu(z), min=-10, max=10)

        return h_a, mu_bu, logvar_bu


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
        """Initialize the AntennaDecoder with deconvolutional layers and a linear head.

        Arguments:
            latent_feat_shape (tuple): The shape of the latent features after convolution.
            flat_dim (int): The flattened dimension of the latent features.
            n_subcarriers (int): The number of subcarriers in the CSI output.
            n_gaussians (int): The dimensionality of the latent space.
            conv_layers (list[tuple[int, int]]): A list of tuples specifying
                the convolutional layers (kernel size and stride) used in the encoder.

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
        z = self._fc(z)
        z = z.view(z.size(0), *self._latent_feat_shape).contiguous()
        z = self._deconv(z)

        return z.squeeze(-1).permute(0, 2, 1)


class Autoencoder(nn.Module):
    """Gaussian autoencoder wrapper exposing hierarchical elements."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_gaussians: int,
        conv_layers: list[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize the Gaussian autoencoder with encoder and decoder components.

        Arguments:
            window_size (int): The size of the time window for CSI input.
            n_subcarriers (int): The number of subcarriers in the CSI input.
            n_gaussians (int): The dimensionality of the latent space.
            conv_layers (list[tuple[int, int]] | None): A list of tuples specifying
                the convolutional layers (kernel size and stride). If None, default layers are used.

        """
        super().__init__()

        if conv_layers is None:
            conv_layers = [(5, 5), (5, 5)]

        self._encoder = _Encoder(window_size, n_subcarriers, n_gaussians, conv_layers)
        latent_feat_shape, flat_dim = self._encoder.get_shapes()
        self._decoder = _Decoder(latent_feat_shape, flat_dim, n_subcarriers, n_gaussians, conv_layers)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_shapes(self) -> tuple[tuple, int]:
        """Get the shapes of the latent features and flattened dimension.

        Returns:
            latent_feat_shape (tuple): The shape of the latent features after convolution.
            flat_dim (int): The flattened dimension of the latent features.

        """
        return self._encoder.get_shapes()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the input CSI window into bottom-up features and distributions.

        Arguments:
            x (torch.Tensor): Input CSI window of shape (batch_size, n_subcarriers, window_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The encoded features and distribution parameters.

        """
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector back into a CSI window.

        Arguments:
            z (torch.Tensor): Latent vector of shape (batch_size, n_gaussians).

        Returns:
            torch.Tensor: Reconstructed CSI window of shape (batch_size, n_subcarriers, window_size).

        """
        return self._decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Arguments:
            x (torch.Tensor): Input CSI window of shape (batch_size, n_subcarriers, window_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Reconstructed CSI window, bottom-up features, mean, and log variance of the latent distribution.

        """
        h_a, mu_bu, logvar_bu = self.encode(x)
        z = self._reparameterize(mu_bu, logvar_bu)
        x_recon = self.decode(z)

        return x_recon, h_a, mu_bu, logvar_bu
