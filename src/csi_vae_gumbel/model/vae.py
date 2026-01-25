import torch
from torch import nn

from csi_vae_gumbel.model.decoder import CSIDecoder
from csi_vae_gumbel.model.encoder import CSIEncoder


class VAE(nn.Module):
    """Variational Autoencoder for CSI data."""

    def __init__(
        self,
        enc_input_shape: tuple[int, int, int] = (450, 2048, 1),
        dec_input_shape: tuple[int, int, int] = (9, 8, 32),
        latent_dim: int = 2,
        categorical_dim: int = 2,
    ) -> None:
        """Initialize the VAE model.

        Arguments:
            enc_input_shape (tuple[int, int, int]): Shape of the encoder input.
            dec_input_shape (tuple[int, int, int]): Shape of the decoder input.
            latent_dim (int): Dimensionality of the latent space.
            categorical_dim (int): Number of categories for the categorical latent variables.

        """
        super().__init__()

        self.encoder = CSIEncoder(enc_input_shape, latent_dim, categorical_dim)
        self.decoder = CSIDecoder(dec_input_shape, latent_dim, categorical_dim, enc_input_shape[-1])

    def forward(
        self,
        x: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE."""
        z, _ = self.encoder(x, tau)
        return self.decoder(z), z
