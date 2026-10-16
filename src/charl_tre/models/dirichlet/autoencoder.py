import torch
from torch import nn


class _Encoder(nn.Module):
    """Encode a single-antenna CSI window into concentration parameters (alpha) for a Dirichlet distribution."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_components: int,
        conv_layers: list[tuple[int, int]],
    ) -> None:
        """Initialize the DirichletEncoder with convolutional layers and linear head.

        Arguments:
            window_size (int): The size of the time window for CSI input.
            n_subcarriers (int): The number of subcarriers in the CSI input.
            n_components (int): The dimensionality of the Dirichlet distribution (number of components).
            conv_layers (list[tuple[int, int]]): A list of tuples specifying
                the convolutional layers (kernel size and stride).

        """
        super().__init__()
        self._window_size = window_size
        self._n_subcarriers = n_subcarriers

        layers: list[nn.Module] = []
        for kh, sh in conv_layers:
            layers.append(nn.Conv2d(n_subcarriers, n_subcarriers, kernel_size=(kh, 1), stride=(sh, 1), bias=False))
            layers.append(nn.BatchNorm2d(n_subcarriers))
            layers.append(nn.GELU())

        layers.append(nn.Flatten())
        self._conv = nn.Sequential(*layers)

        # Infer flattened feature dimension for linear head
        _, flat_dim = self.get_shapes()

        #  Linear head to produce concentration parameters (alpha) for the Dirichlet distribution
        self._alpha = nn.Sequential(nn.Linear(flat_dim, n_components), nn.Softplus())

    @torch.no_grad()
    def get_shapes(self) -> tuple[tuple, int]:
        """Return the latent feature map shape and its flattened size.

        Returns:
            latent_feat_shape (tuple[int, int, int]): The shape of the feature map after convolution (Channels, H, W).
            flat_dim (int): The total number of features when the feature map is flattened.

        """
        x = torch.zeros(1, self._n_subcarriers, self._window_size, 1, device=next(self.parameters()).device)
        x = self._conv[:-1](x)
        return x.shape[1:], int(x.numel() // x.shape[0])

    def forward(self, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Compute concentration parameters (alpha) for a single-antenna input.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, n_subcarriers) for one antenna.
            eps (float): Small constant to add to the output to avoid zero concentration parameters.

        Returns:
            alpha (torch.Tensor): Tensor of shape (batch_size, n_components) representing
                the concentration parameters of the Dirichlet distribution.

        """
        x = x.permute(0, 2, 1).unsqueeze(-1).contiguous()  # (batch_size, n_subcarriers, window_size, 1)
        z = self._conv(x)

        # Add a small constant to avoid zero concentration parameters,
        # which can cause numerical issues in the Dirichlet distribution.
        return self._alpha(z) + eps


class _Decoder(nn.Module):
    """Decode a latent vector back into a CSI window for a single antenna."""

    def __init__(
        self,
        latent_feat_shape: tuple,
        flat_dim: int,
        n_subcarriers: int,
        n_components: int,
        conv_layers: list[tuple[int, int]],
    ) -> None:
        """Initialize the DirichletDecoder with linear and deconvolutional layers.

        Arguments:
            latent_feat_shape (tuple[int, int, int]): The shape of the feature map before flattening in the encoder.
            flat_dim (int): The total number of features when the feature map is flattened.
            n_subcarriers (int): The number of subcarriers in the CSI input.
            n_components (int): The number of Dirichlet components to decode from.
            conv_layers (list[tuple[int, int]]): A list of tuples
                specifying the convolutional layers (kernel size and stride)

        """
        super().__init__()
        self._latent_feat_shape = latent_feat_shape

        self._fc = nn.Sequential(nn.Linear(n_components, flat_dim), nn.GELU())

        deconv_layers: list[nn.Module] = []
        reversed_specs = list(reversed(conv_layers))

        for i, (kh, sh) in enumerate(reversed_specs):
            deconv_layers.append(
                nn.ConvTranspose2d(
                    n_subcarriers,
                    n_subcarriers,
                    kernel_size=(kh, 1),
                    stride=(sh, 1),
                    bias=i >= len(reversed_specs) - 1,  # No bias for intermediate layers
                ),
            )
            if i < len(reversed_specs) - 1:
                deconv_layers.append(nn.BatchNorm2d(n_subcarriers))
                deconv_layers.append(nn.GELU())

        self._deconv = nn.Sequential(*deconv_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector into a CSI window.

        Arguments:
            z (torch.Tensor): Input tensor of shape (batch_size, n_components)
                representing the latent vector for one antenna.

        Returns:
            recon (torch.Tensor): Tensor of shape (batch_size, window_size, n_subcarriers)
                representing the reconstructed CSI window for one antenna.

        """
        z = self._fc(z)  # (batch_size, flat_dim)
        z = z.view(z.size(0), *self._latent_feat_shape).contiguous()  # (batch_size, n_subcarriers, window_size', 1)
        z = self._deconv(z)  # (batch_size, n_subcarriers, window_size, 1)
        return z.squeeze(-1).permute(0, 2, 1)  # (batch_size, window_size, n_subcarriers)


class Autoencoder(nn.Module):
    """Dirichlet VAE architecture that encodes CSI data."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_components: int,
        conv_layers: list[tuple[int, int]],
    ) -> None:
        """Initialize the DirichletVAE with an encoder and decoder for CSI data.

        Arguments:
            window_size (int): The size of the time window for CSI input.
            n_subcarriers (int): The number of subcarriers in the CSI input.
            n_components (int): The number of components in the Dirichlet distribution (latent space dimensionality).
            conv_layers (list[tuple[int, int]]): A list of tuples
                specifying the convolutional layers (kernel size and stride).

        """
        super().__init__()

        self._encoder = _Encoder(window_size, n_subcarriers, n_components, conv_layers)
        latent_feat_shape, flat_dim = self._encoder.get_shapes()
        self._decoder = _Decoder(latent_feat_shape, flat_dim, n_subcarriers, n_components, conv_layers)
        self._n_components = n_components

        with torch.no_grad():
            dummy = torch.zeros(2, window_size, n_subcarriers)
            recon, _ = self.forward(dummy)
            if recon.shape != dummy.shape:
                msg = f"Decoder output shape {recon.shape} does not match input shape {dummy.shape}"
                raise ValueError(msg)

    @staticmethod
    def reparameterize(alpha: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for Dirichlet distribution.

        Arguments:
            alpha (torch.Tensor): Concentration parameters of shape (batch_size, n_components)

        Returns:
            z (torch.Tensor): Sampled latent variable of shape (batch_size, n_components)

        """
        dirichlet_dist = torch.distributions.Dirichlet(alpha)
        return dirichlet_dist.rsample()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input CSI window into concentration parameters.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, n_subcarriers)

        Returns:
            alpha (torch.Tensor): Tensor of shape (batch_size, n_components)
                representing the concentration parameters of the Dirichlet distribution.

        """
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector to reconstruct the input.

        Arguments:
            z (torch.Tensor): Input tensor of shape (batch_size, n_components)

        Returns:
            recon (torch.Tensor): Tensor of shape (batch_size, window_size, n_subcarriers)
                representing the reconstructed CSI window.

        """
        return self._decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the input, sample a latent variable, and decode to reconstruct the input.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, window_size, n_subcarriers).

        Returns:
            recon (torch.Tensor): Tensor of shape (batch_size, window_size, n_subcarriers)
                representing the reconstructed input.
            alpha (torch.Tensor): Tensor of shape (batch_size, n_components)
                representing the concentration parameters of the Dirichlet distribution.

        """
        alpha = self.encode(x)
        z = self.reparameterize(alpha)
        recon = self.decode(z)

        return recon, alpha
