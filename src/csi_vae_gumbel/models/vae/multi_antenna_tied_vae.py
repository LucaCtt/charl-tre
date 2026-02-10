import torch
from torch import nn
from torch.nn import functional as func


class MultiAntennaTiedVAE(nn.Module):
    """Multi-View Categorical VAE for CSI data from multiple antennas."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_antennas: int,
        n_categories: int,
        latent_dim: int,
    ) -> None:
        """Initialize the Multi-View Categorical VAE.

        Arguments:
            window_size (int): Size of the time window.
            n_subcarriers (int): Number of subcarriers.
            n_antennas (int): Number of antennas (views).
            n_categories (int): Number of categories for the categorical latent variables.
            latent_dim (int): Dimension of each categorical variable.

        """
        super().__init__()

        self.__window_size = window_size
        self.__n_subcarriers = n_subcarriers
        self.__n_antennas = n_antennas
        self.__n_categories = n_categories
        self.__latent_dim = latent_dim

        # Convolutional weights for each antenna
        # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        self.__conv_weights_1 = nn.ParameterList(
            [nn.Parameter(torch.empty(8, 1, 3, 3)) for _ in range(n_antennas)],
        )
        self.__enc_conv_biases_1 = nn.ParameterList([nn.Parameter(torch.zeros(8)) for _ in range(n_antennas)])
        self.__dec_conv_biases_1 = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_antennas)])

        self.__conv_weights_2 = nn.ParameterList(
            [nn.Parameter(torch.empty(16, 8, 3, 3)) for _ in range(n_antennas)],
        )
        self.__enc_conv_biases_2 = nn.ParameterList([nn.Parameter(torch.zeros(16)) for _ in range(n_antennas)])
        self.__dec_conv_biases_2 = nn.ParameterList([nn.Parameter(torch.zeros(8)) for _ in range(n_antennas)])

        # Compute shapes dynamically
        self.__latent_feat_shape, self.__out_paddings = self.__get_shapes_and_paddings()
        self.__flat_dim = int(torch.prod(torch.tensor(self.__latent_feat_shape)).item())

        # Per-antenna linear weights
        self.lin_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(latent_dim * n_categories, self.__flat_dim)) for _ in range(n_antennas)],
        )
        self.lin_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(latent_dim * n_categories)) for _ in range(n_antennas)],
        )

        # Central bottleneck weights, maps n_antennas latents -> combined latent
        self.bottleneck_enc = nn.Linear(latent_dim * n_categories * n_antennas, latent_dim * n_categories)
        self.bottleneck_dec = nn.Linear(latent_dim * n_categories, latent_dim * n_categories * n_antennas)

        self.__init_weights()

    def __init_weights(self) -> None:
        """Initialize weights for the model."""
        # Kaiming initialization preserves variance through ReLU layers
        for w in [*self.__conv_weights_1, *self.__conv_weights_2, *self.lin_weights]:
            nn.init.kaiming_normal_(w, nonlinearity="relu")

    def __get_shapes_and_paddings(self) -> tuple:
        """Mock pass to find flattened size and required output_paddings."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.__window_size, self.__n_subcarriers)

            # Trace Layer 1
            l1 = func.conv2d(x, self.__conv_weights_1[0], self.__enc_conv_biases_1[0], stride=2, padding=1)
            # Trace Layer 2
            l2 = func.conv2d(l1, self.__conv_weights_2[0], self.__enc_conv_biases_2[0], stride=2, padding=1)

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs from multiple antennas.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers).

        Returns:
            torch.Tensor: Concatenated encoded latent tensor of shape (batch_size, latent_per_antenna * n_antennas).

        """
        encoded_latents = []

        for i in range(self.__n_antennas):
            zi = x[:, i : i + 1, :, :]  # Shape: (batch_size, 1, window_size, n_subcarriers)

            # First convolutional layer
            zi = func.conv2d(zi, self.__conv_weights_1[i], self.__enc_conv_biases_1[i], stride=2, padding=1)
            zi = func.relu(zi)

            # Second convolutional layer
            zi = func.conv2d(zi, self.__conv_weights_2[i], self.__enc_conv_biases_2[i], stride=2, padding=1)
            zi = func.relu(zi)

            # Flatten
            zi = zi.view(zi.size(0), -1)

            # linear layer to get antenna latent
            zi = func.linear(zi, self.lin_weights[i], self.lin_biases[i])

            encoded_latents.append(zi)

        z = torch.cat(encoded_latents, dim=1)
        return self.bottleneck_enc(z)

    def decode(self, z_hard: torch.Tensor) -> torch.Tensor:
        """Decode from categorical latent back to multiple antennas.

        Arguments:
            z_hard (torch.Tensor): Hard categorical latent tensor of shape
                                   (batch_size, latent_dim * n_categories).

        Returns:
            torch.Tensor: Reconstructed outputs of shape (batch_size, n_antennas, window_size, n_subcarriers).

        """
        recon_combined = self.bottleneck_dec(z_hard)
        recon_latents = torch.chunk(recon_combined, self.__n_antennas, dim=1)

        recons = []
        for i, zi in enumerate(recon_latents):
            # Linear layer to expand back to conv feature map size
            xi = func.linear(zi, self.lin_weights[i].t())
            xi = func.relu(xi)

            # Reshape to conv feature map
            xi = xi.view(-1, *self.__latent_feat_shape)

            # First transposed convolutional layer
            xi = func.conv_transpose2d(
                xi,
                self.__conv_weights_2[i],
                self.__dec_conv_biases_2[i],
                stride=2,
                padding=1,
                output_padding=self.__out_paddings[0],
            )
            xi = func.relu(xi)

            xi = func.conv_transpose2d(
                xi,
                self.__conv_weights_1[i],
                self.__dec_conv_biases_1[i],
                stride=2,
                padding=1,
                output_padding=self.__out_paddings[1],
            )

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
        logits = self.encode(x)
        logits = logits.view(-1, self.__latent_dim, self.__n_categories)

        # Gumbel-Softmax sampling
        z_hard = func.gumbel_softmax(logits, tau=tau, hard=True)

        # Map back to combined latent space
        z_hard_flat = z_hard.view(z_hard.size(0), -1)
        recon = self.decode(z_hard_flat)

        return recon, z_hard, logits
