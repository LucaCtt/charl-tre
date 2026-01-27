import torch
from torch import nn


class MultiViewCategoricalVAE(nn.Module):
    """Multi-View Categorical VAE for CSI data from multiple antennas."""

    def __init__(
        self,
        window_size: int,
        n_subcarriers: int,
        n_antennas: int,
        hidden_latent_dim: int,
        n_categories: int,
        categorical_dim: int,
    ) -> None:
        """Initialize the Multi-View Categorical VAE.

        Arguments:
            window_size (int): Size of the time window.
            n_subcarriers (int): Number of subcarriers.
            n_antennas (int): Number of antennas (views).
            hidden_latent_dim (int): Dimension of the latent space per antenna.
            n_categories (int): Number of categories for the categorical latent variables.
            categorical_dim (int): Dimension of each categorical variable.

        """
        super().__init__()

        self.window_size = window_size
        self.n_subcarriers = n_subcarriers
        self.n_antennas = n_antennas
        self.n_categories = n_categories
        self.categorical_dim = categorical_dim

        # Separate encoder weights for each of the antennas.
        self.enc_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_latent_dim, window_size * n_subcarriers)) for _ in range(n_antennas)],
        )
        for w in self.enc_weights:
            # Kaiming initialization preserves variance through ReLU layers
            nn.init.kaiming_normal_(w, nonlinearity="relu")
        self.enc_biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden_latent_dim)) for _ in range(n_antennas)])

        # Central bottleneck weights, maps n_antennas latents -> Categorical Logits)
        # Total input to bottleneck = latent_dim * n_antennas
        self.bottleneck_weight = nn.Parameter(
            torch.empty(n_categories * categorical_dim, hidden_latent_dim * n_antennas),
        )
        # Xavier is better for layers that don't use ReLU,
        # because it keeps the initial logits near zero for a uniform Gumbel start.
        # This helps preventing the model from collapsing to a single category early in training.
        nn.init.xavier_uniform_(self.bottleneck_weight)
        self.bottleneck_bias = nn.Parameter(torch.zeros(n_categories * categorical_dim))

    def encode(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Encode inputs from multiple antennas.

        Arguments:
            xs (list[torch.Tensor]): List of input tensors from each antenna,
                                     each of shape (batch_size, window_size * n_subcarriers).

        Returns:
            torch.Tensor: Concatenated encoded latent tensor of shape (batch_size, latent_dim * n_antennas).

        """
        encoded_latents = []
        for i in range(self.n_antennas):
            # Tied weight linear: x @ W.T + b
            z = nn.functional.linear(xs[i], self.enc_weights[i], self.enc_biases[i])
            z = nn.functional.relu(z)
            encoded_latents.append(z)
        return torch.cat(encoded_latents, dim=1)

    def decode(self, z_cat_flat: torch.Tensor) -> list[torch.Tensor]:
        """Decode from categorical latent back to multiple antennas.

        Arguments:
            z_cat_flat (torch.Tensor): Flattened categorical latent tensor of shape
                                       (batch_size, categorical_dim * num_categories).

        Returns:
            list[torch.Tensor]: List of reconstructed tensors for each antenna,

        """
        # Mirror of bottleneck: Use transposed weight
        combined_latent = nn.functional.linear(z_cat_flat, self.bottleneck_weight.t())
        combined_latent = nn.functional.relu(combined_latent)

        # Split back into n_antennas separate latent spaces
        latents = torch.chunk(combined_latent, self.n_antennas, dim=1)

        # Pass through mirrored encoders to get n_antennas reconstructions
        recons = []
        for i in range(self.n_antennas):
            # Tied weight decode: Use the same weight tensor from the encoder
            # We don't reuse the encoder bias for the output layer because it hurts performance
            r = nn.functional.linear(latents[i], self.enc_weights[i].t())
            recons.append(torch.sigmoid(r))
        return recons

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Multi-View Categorical VAE.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers)
            tau (float): Temperature parameter for Gumbel-Softmax.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Reconstructed outputs and logits.

        """
        if x.size() != (x.size(0), self.n_antennas, self.window_size, self.n_subcarriers):
            msg = (
                f"Expected input shape {(x.size(0), self.n_antennas, self.window_size, self.n_subcarriers)}, "
                f"got {x.size()}"
            )
            raise ValueError(msg)

        # Split into list of antennas
        x_split = torch.split(x, 1, dim=1)

        # Flatten each antenna input
        xs = [xi.view(xi.size(0), -1) for xi in x_split]

        # Encoder -> combined latent
        combined_z = self.encode(xs)

        # Map to Categorical Logits
        logits = nn.functional.linear(combined_z, self.bottleneck_weight, self.bottleneck_bias)
        logits = logits.view(-1, self.categorical_dim, self.n_categories)

        # Gumbel-Softmax Sampling
        z_cat = nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        z_cat_flat = z_cat.view(z_cat.size(0), -1)

        # Decode and stack
        reconstructions = self.decode(z_cat_flat)
        reconstruction = torch.stack(reconstructions, dim=1)

        return reconstruction, logits
