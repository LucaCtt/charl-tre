import torch
import torch.nn.functional as func
from torch import nn

from csi_vae_gumbel.models.vae.single_antenna_vae import SingleAntennaVAE


class MultiAntennaVAE(nn.Module):
    """Multi-antenna categorical VAE with Gumbel-Softmax reparameterization."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_antennas: int,
        n_categories: int,
        latent_dim: int,
    ) -> None:
        """Initialize the multi-antenna VAE model.

        Arguments:
            window_size: Size of the time window in the input data.
            n_subcarriers: Number of subcarriers in the input data.
            n_antennas: Number of antennas (views) in the input data.
            n_categories: Number of categories for the categorical latent variables.
            latent_dim: Dimensionality of the latent space.

        """
        super().__init__()
        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers
        self.__n_antennas = n_antennas
        self.__n_categories = n_categories
        self.__latent_dim = latent_dim

        self.__antenna_vaes = nn.ModuleList(
            [SingleAntennaVAE(window_size, n_subcarriers, n_categories, latent_dim) for _ in range(n_antennas)],
        )

        self.__encoder_bottleneck = nn.Linear(n_antennas * n_categories * latent_dim, n_categories * latent_dim)

        self.__decoder_bottleneck = nn.Linear(n_categories * latent_dim, n_antennas * n_categories * latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode input into logits for categorical distribution.

        Arguments:
            x: Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers)

        Returns:
            Logits tensor of shape (batch_size, n_antennas * latent_dim * n_categories)
            The same logits but in list form, each item of shape (batch_size, latent_dim * n_categories)

        """
        logits = []
        for i, vae in enumerate(self.__antenna_vaes):
            logit = x[:, i : i + 1, :, :]  # Shape: (batch_size, 1, window_size, n_subcarriers)
            logit = vae.encode(logit)  # pyright: ignore[reportCallIssue]
            logits.append(logit)

        return torch.cat(logits, dim=1), logits

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode latent representation back to input space.

        Arguments:
            z: Latent tensor of shape (batch_size, latent_dim * n_categories)

        Returns:
            Reconstructed tensor of shape (batch_size, 1, window_size, n_subcarriers)
            The latents input to each antenna's decoder, in list form,
            each item of shape (batch_size, latent_dim * n_categories)

        """
        recon = []
        latents_input = []
        for i, vae in enumerate(self.__antenna_vaes):
            zi = z[:, i * self.__n_categories * self.__latent_dim : (i + 1) * self.__n_categories * self.__latent_dim]
            latents_input.append(zi)
            zi = vae.decode(zi)  # pyright: ignore[reportCallIssue]
            recon.append(zi)

        return torch.cat(recon, dim=1), latents_input

    def forward(
        self,
        x: torch.Tensor,
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through the VAE.

        Arguments:
            x: Input tensor of shape (batch_size, 1, window_size, n_subcarriers)
            tau: Temperature parameter for Gumbel-Softmax

        Returns:
            recon: Reconstructed tensor of shape (batch_size, 1, window_size, n_subcarriers)
            z_hard: One-hot latent tensor of shape (batch_size, latent_dim, n_categories)
            logits: Logits tensor of shape (batch_size, latent_dim, n_categories)
            encoder_latents: List of intermediate logits for each antenna
            decoder_recons: List of intermediate reconstructions for each antenna

        """
        logits, logits_per_antenna = self.encode(x)

        # Map to categorical logits
        logits = self.__encoder_bottleneck(logits)

        # Gumbel-Softmax sampling
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True)

        # Map back to combined latent space
        recon = self.__decoder_bottleneck(z_hard)

        # Decode and stack
        recon, latents_input = self.decode(recon)

        return recon, z_hard, logits, logits_per_antenna, latents_input
