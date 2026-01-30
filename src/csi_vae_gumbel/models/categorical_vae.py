import torch
from torch import nn
from torch.nn import functional as func


class CategoricalVAE(nn.Module):
    """Categorical VAE for CSI data from a single antenna."""

    def __init__(
        self,
        window_size: int,
        n_categories: int,
        categorical_dim: int,
    ) -> None:
        """Initialize Categorical VAE.

        Arguments:
            window_size (int): Size of the time window.
            n_categories (int): Number of latent categorical variables.
            categorical_dim (int): Number of classes per categorical variable.

        """
        super().__init__()

        self.window_size = window_size
        self.n_categories = n_categories
        self.categorical_dim = categorical_dim

        # Encoder: Convolutional layers
        self.encoder_conv_1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.encoder_conv_2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)

        # Latent space: 16 * 38 * 64 is the flattened size after strides
        self.flat_dim = 16 * 38 * 64
        self.encoder_fc = nn.Linear(self.flat_dim, n_categories * categorical_dim)

        # Decoder: Mirror of encoder
        self.decoder_fc = nn.Linear(n_categories * categorical_dim, self.flat_dim)
        self.decoder_deconv_1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=(0, 1))
        self.decoder_deconv_2 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input CSI data into raw logits for categorical distributions.

        Arguments:
            x (torch.Tensor): Input CSI data.

        Returns:
            torch.Tensor: Raw logits for categorical distributions.

        """
        z = func.relu(self.encoder_conv_1(x))
        z = func.relu(self.encoder_conv_2(z))

        z = z.view(z.size(0), -1)
        return self.encoder_fc(z)

    def decode(self, latent_repr: torch.Tensor) -> torch.Tensor:
        """Decode latent categorical representation back to CSI data.

        Arguments:
            latent_repr (torch.Tensor): Latent categorical representation.

        Returns:
            torch.Tensor: Reconstructed CSI data.

        """
        z = func.relu(self.decoder_fc(latent_repr))
        z = z.view(z.size(0), 16, 38, 64)

        z = func.relu(self.decoder_deconv_1(z))

        # Final layer: No ReLU, use Sigmoid to bound CSI between 0 and 1
        return torch.sigmoid(self.decoder_deconv_2(z))

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the Categorical VAE.

        Arguments:
            x (torch.Tensor): Input CSI data.
            tau (float): Temperature parameter for Gumbel-Softmax.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed CSI data,
            hard one-hot latent samples, and raw logits.

        """
        # 1. Get raw logits from encoder
        logits_flat = self.encode(x)

        # 2. Reshape to (Batch, N_Latents, Classes_Per_Latent)
        # PyTorch Gumbel Softmax expects classes on the LAST dimension
        logits = logits_flat.view(-1, self.n_categories, self.categorical_dim)

        # 3. Gumbel-Softmax Reparameterization
        # hard=True returns one-hot during forward, but keeps grads via relaxation
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True)

        # 4. Flatten back for the decoder
        z_cat_flat = z_hard.view(z_hard.size(0), -1)

        # 5. Reconstruct
        recon = self.decode(z_cat_flat)

        return recon, z_hard, logits
