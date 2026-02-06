import torch
import torch.nn.functional as func
from torch import nn


class SingleAntennaVAE(nn.Module):
    """Categorical VAE with Gumbel-Softmax reparameterization."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_categories: int,
        latent_dim: int,
    ) -> None:
        """Initialize the Single Antenna VAE model.

        Arguments:
            window_size: Size of the time window in the input data.
            n_subcarriers: Number of subcarriers in the input data.
            n_categories: Number of categories for the categorical latent variables.
            latent_dim: Dimensionality of the latent space.

        """
        super().__init__()
        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers
        self.__n_categories = n_categories
        self.__latent_dim = latent_dim

        # Encoder group
        self.__encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Dynamic dimension capture
        self.__latent_feat_shape, out_paddings = self.__get_shapes_and_paddings()
        self.__flat_dim = int(torch.prod(torch.tensor(self.__latent_feat_shape)).item())
        self.__encoder_fc = nn.Linear(self.__flat_dim, n_categories * latent_dim)

        # Decoder group
        self.__decoder_fc = nn.Linear(n_categories * latent_dim, self.__flat_dim)
        self.__decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=out_paddings[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=out_paddings[1]),
            nn.Sigmoid(),
        )

    def __get_shapes_and_paddings(self) -> tuple:
        """Mock pass to find flattened size and required output_paddings."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.__window_size, self.__n_subcarriers)

            # Trace Layer 1
            l1 = self.__encoder_conv[0](x)
            # Trace Layer 2
            l2 = self.__encoder_conv[2](l1)

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

    def __encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into logits for categorical distribution.

        Arguments:
            x: Input tensor of shape (batch_size, 1, window_size, n_subcarriers)

        Returns:
            Logits tensor of shape (batch_size, latent_dim * n_categories)

        """
        z = self.__encoder_conv(x)
        return self.__encoder_fc(z.view(z.size(0), -1))

    def __decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space.

        Arguments:
            z: Latent tensor of shape (batch_size, latent_dim * n_categories)

        Returns:
            Reconstructed tensor of shape (batch_size, 1, window_size, n_subcarriers)

        """
        z = func.relu(self.__decoder_fc(z))
        z = z.view(-1, *self.__latent_feat_shape)
        return self.__decoder_conv(z)

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> tuple:
        """Forward pass through the VAE.

        Arguments:
            x: Input tensor of shape (batch_size, 1, window_size, n_subcarriers)
            tau: Temperature parameter for Gumbel-Softmax

        Returns:
            recon: Reconstructed tensor of shape (batch_size, 1, window_size, n_subcarriers)
            z_hard: One-hot latent tensor of shape (batch_size, latent_dim, n_categories)
            logits: Logits tensor of shape (batch_size, latent_dim, n_categories)

        """
        logits_flat = self.__encode(x)
        logits = logits_flat.view(-1, self.__latent_dim, self.__n_categories)
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)

        recon = self.__decode(z_hard.view(z_hard.size(0), -1))
        return recon, z_hard, logits
