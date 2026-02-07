import torch
import torch.nn.functional as func
from torch import nn


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

        # Encoder group
        self.__encoder_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                )
                for _ in range(n_antennas)
            ],
        )
        # Dynamic dimension capture
        self.__latent_feat_shape, out_paddings = self.__get_shapes_and_paddings()
        flat_dim = int(torch.prod(torch.tensor(self.__latent_feat_shape)).item())

        self.__encoder_fcs = nn.ModuleList(
            [nn.Linear(flat_dim, n_categories * latent_dim) for _ in range(n_antennas)],
        )

        self.__encoder_bottleneck = nn.Linear(n_antennas * n_categories * latent_dim, n_categories * latent_dim)

        self.__decoder_bottleneck = nn.Linear(n_categories * latent_dim, n_antennas * n_categories * latent_dim)

        # Decoder group
        self.__decoder_fcs = nn.ModuleList(
            [nn.Linear(n_categories * latent_dim, flat_dim) for _ in range(n_antennas)],
        )
        self.__decoder_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=out_paddings[0]),
                    nn.ReLU(),
                    nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=out_paddings[1]),
                    nn.Sigmoid(),
                )
                for _ in range(n_antennas)
            ],
        )

    def __get_shapes_and_paddings(self) -> tuple:
        """Mock pass to find flattened size and required output_paddings."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.__window_size, self.__n_subcarriers)

            # Trace Layer 1
            l1 = self.__encoder_convs[0][0](x)  # pyright: ignore[reportIndexIssue]
            # Trace Layer 2
            l2 = self.__encoder_convs[0][2](l1)  # pyright: ignore[reportIndexIssue]

            # Helper to find output_padding needed for a specific layer
            def __get_op(in_shape: int, out_target: int) -> int:
                # Standard formula: out = (in-1)*s - 2*p + k
                current_out = (in_shape - 1) * 2 - 2 * 1 + 3
                return out_target - current_out

            # op1: From l2 back to l1
            op1 = (__get_op(l2.shape[2], l1.shape[2]), __get_op(l2.shape[3], l1.shape[3]))
            # op2: From l1 back to original x
            op2 = (__get_op(l1.shape[2], x.shape[2]), __get_op(l1.shape[3], x.shape[3]))

            return l2.shape[1:], (op1, op2)

    def __encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode input into logits for categorical distribution.

        Arguments:
            x: Input tensor of shape (batch_size, 1, window_size, n_subcarriers)

        Returns:
            Logits tensor of shape (batch_size, latent_dim * n_categories)
            List of intermediate logits for each antenna

        """
        logits = []
        for i in range(self.__n_antennas):
            xi = x[:, i : i + 1, :, :]  # Shape: (batch_size, 1, window_size, n_subcarriers)
            xi = self.__encoder_convs[i](xi)  # pyright: ignore[reportIndexIssue]
            xi = xi.view(xi.size(0), -1)
            xi = self.__encoder_fcs[i](xi)  # pyright: ignore[reportIndexIssue]
            logits.append(xi)

        return torch.cat(logits, dim=1), logits

    def __decode(self, z: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Decode latent representation back to input space.

        Arguments:
            z: Latent tensor of shape (batch_size, latent_dim * n_categories)

        Returns:
            Reconstructed tensor of shape (batch_size, 1, window_size, n_subcarriers)
            List of intermediate reconstructions for each antenna

        """
        recon = []
        for i in range(self.__n_antennas):
            zi = z[:, i * self.__n_categories * self.__latent_dim : (i + 1) * self.__n_categories * self.__latent_dim]
            zi = self.__decoder_fcs[i](zi)  # pyright: ignore[reportIndexIssue]
            zi = zi.view(-1, *self.__latent_feat_shape)
            zi = self.__decoder_convs[i](zi)  # pyright: ignore[reportIndexIssue]
            recon.append(zi)

        return torch.cat(recon, dim=1), recon

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
        # Encoder -> combined latent
        combined_latents, encoder_latents = self.__encode(x)

        # Map to categorical logits
        logits = self.__encoder_bottleneck(combined_latents)

        # Gumbel-Softmax sampling
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True)

        # Map back to combined latent space
        recon_combined = self.__decoder_bottleneck(z_hard)

        # Decode and stack
        recon, decoder_recons = self.__decode(recon_combined)

        return recon, z_hard, logits, encoder_latents, decoder_recons
