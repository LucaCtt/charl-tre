"""CSI encoder module for VAE."""

import torch
from torch import nn

from csi_vae_gumbel.model.gumbel import gumbel_softmax_straight_through_stable


class CSIEncoder(nn.Module):
    """CSI encoder module for VAE."""

    def __init__(self, input_shape: tuple[int, int, int], latent_dim: int, categorical_dim: int) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=(5, 8), stride=(5, 8)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(5, 8), stride=(5, 8)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(2, 4), stride=(2, 4)),
            nn.ReLU(),
        )

        # Infer flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            flat_dim = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 24),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        tau: float,
        eps_u: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the encoder."""
        z = self.conv(x)
        z = self.fc(z)
        z = z.view(-1, self.latent_dim, self.categorical_dim)

        # Ensure float and pre-center logits to reduce magnitude (safe due to softmax shift-invariance)
        z = z.float()
        z = z - z.detach().amax(dim=-1, keepdim=True)

        # Straight-Through Gumbel-Softmax: hard forward, soft gradient
        z_gumbel = gumbel_softmax_straight_through_stable(
            z,
            tau=tau,
            dim=-1,
            eps_u=eps_u,
            center=False,  # already centered above
        )

        # Flatten [B, D, K] -> [B, D*K] for the latent vector
        z = z_gumbel.view(-1, self.latent_dim * self.categorical_dim)
        return z, z_gumbel
