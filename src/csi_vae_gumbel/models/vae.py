import torch
from torch import nn
from torch.nn import functional as func


class MultiViewCategoricalVAE(nn.Module):
    """Multi-View Categorical VAE for CSI data from multiple antennas."""

    def __init__(
        self,
        window_size: int,
        n_antennas: int,
        n_categories: int,
        categorical_dim: int,
        hidden_latent_dim: int,
    ) -> None:
        """Initialize the Multi-View Categorical VAE.

        Arguments:
            window_size (int): Size of the time window.
            n_antennas (int): Number of antennas (views).
            n_categories (int): Number of categories for the categorical latent variables.
            categorical_dim (int): Dimension of each categorical variable.
            hidden_latent_dim (int): Dimension of the hidden latent space per antenna.

        """
        super().__init__()

        self.window_size = window_size
        self.n_antennas = n_antennas
        self.n_categories = n_categories
        self.categorical_dim = categorical_dim

        # Tied convulutional weights for each antenna
        # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        self.conv_weights_1 = nn.ParameterList(
            [nn.Parameter(torch.empty(8, 1, 3, 3)) for _ in range(n_antennas)],
        )
        self.conv_biases_1 = nn.ParameterList([nn.Parameter(torch.zeros(8)) for _ in range(n_antennas)])

        self.conv_weights_2 = nn.ParameterList(
            [nn.Parameter(torch.empty(16, 8, 3, 3)) for _ in range(n_antennas)],
        )
        self.conv_biases_2 = nn.ParameterList([nn.Parameter(torch.zeros(16)) for _ in range(n_antennas)])

        self.feat_h, self.feat_w = 38, 64
        self.conv_flat_dim = 16 * self.feat_h * self.feat_w

        # Per-antenna tied linear weights
        self.lin_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_latent_dim, self.conv_flat_dim)) for _ in range(n_antennas)],
        )
        self.lin_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_latent_dim)) for _ in range(n_antennas)],
        )

        # Central bottleneck weights, maps n_antennas latents -> Categorical Logits)
        # Total input to bottleneck = latent_dim * n_antennas
        self.bottleneck_weight = nn.Parameter(
            torch.empty(n_categories * categorical_dim, hidden_latent_dim * n_antennas),
        )
        self.bottleneck_bias = nn.Parameter(torch.zeros(n_categories * categorical_dim))

        # Learnable scale for the tied decoder,
        # because the optimal magnitude for decoding may differ from encoding.
        # Initialized to 1.0 so it doesn't disrupt training at the start
        self.decoder_scale = nn.Parameter(torch.ones(n_antennas))

        self.__init_weights()

    def __init_weights(self) -> None:
        """Initialize weights for the model."""
        # Kaiming initialization preserves variance through ReLU layers
        for w in [*self.conv_weights_1, *self.conv_weights_2, *self.lin_weights]:
            nn.init.kaiming_normal_(w, nonlinearity="relu")

        # Xavier is better for layers that don't use ReLU,
        # because it keeps the initial logits near zero for a uniform Gumbel start.
        # This helps preventing the model from collapsing to a single category early in training.
        nn.init.xavier_uniform_(self.bottleneck_weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs from multiple antennas.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers).

        Returns:
            torch.Tensor: Concatenated encoded latent tensor of shape (batch_size, latent_per_antenna * n_antennas).

        """
        encoded_latents = []

        for i in range(self.n_antennas):
            zi = x[:, i : i + 1, :, :]  # Shape: (batch_size, 1, window_size, n_subcarriers)

            # First convolutional layer
            zi = func.conv2d(zi, self.conv_weights_1[i], self.conv_biases_1[i], stride=2, padding=1)
            zi = func.relu(zi)

            # Second convolutional layer
            zi = func.conv2d(zi, self.conv_weights_2[i], self.conv_biases_2[i], stride=2, padding=1)
            zi = func.relu(zi)

            # Flatten
            zi = zi.view(zi.size(0), -1)

            # linear layer to get antenna latent
            zi = func.linear(zi, self.lin_weights[i], self.lin_biases[i])
            zi = func.relu(zi)

            encoded_latents.append(zi)

        return torch.cat(encoded_latents, dim=1)

    def decode(self, recon_combined: torch.Tensor) -> torch.Tensor:
        """Decode from categorical latent back to multiple antennas.

        Arguments:
            recon_combined (torch.Tensor): Combined latent tensor of shape
                                           (batch_size, latent_per_antenna * n_antennas).

        Returns:
            torch.Tensor: Reconstructed outputs of shape (batch_size, n_antennas, window_size, n_subcarriers).

        """
        recon_latents = torch.chunk(recon_combined, self.n_antennas, dim=1)

        recons = []
        for i, zi in enumerate(recon_latents):
            # Linear layer to expand back to conv feature map size
            xi = func.linear(zi, self.lin_weights[i].t())
            xi = func.relu(xi)

            # Reshape to conv feature map
            xi = xi.view(xi.size(0), 16, 38, 64)

            # First transposed convolutional layer
            xi = func.conv_transpose2d(xi, self.conv_weights_2[i], stride=2, padding=1, output_padding=(0, 1))
            xi = func.relu(xi)

            xi = func.conv_transpose2d(xi, self.conv_weights_1[i], stride=2, padding=1, output_padding=(1, 1))

            # No ReLU here otherwise otherwise negative values become 0, sigmoid(0) = 0.5
            # so we cannot reconstruct values near 0 properly.

            # Final sigmoid activation with learnable scale
            xi = torch.sigmoid(xi * self.decoder_scale[i])

            recons.append(xi)
        return torch.stack(recons, dim=1).squeeze(2)

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the Multi-View Categorical VAE.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers)
            tau (float): Temperature parameter for Gumbel-Softmax.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed output tensor,
                                                             hard categorical latent tensor,
                                                             and logits tensor.

        """
        # Encoder -> combined latent
        combined_z = self.encode(x)

        # Map to categorical logits
        logits = func.linear(combined_z, self.bottleneck_weight, self.bottleneck_bias)
        logits = func.relu(logits)
        logits = logits.view(-1, self.categorical_dim, self.n_categories)

        # Gumbel-Softmax sampling
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True)

        # Map back to combined latent space
        z_cat_flat = z_hard.view(z_hard.size(0), -1)
        recon_combined = func.linear(z_cat_flat, self.bottleneck_weight.t())
        recon_combined = func.relu(recon_combined)

        # Decode and stack
        recon = self.decode(recon_combined)

        return recon, z_hard, logits
