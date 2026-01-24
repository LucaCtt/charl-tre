"""CSI decoder module for VAE."""

import torch
from torch import nn


class CSIDecoder(nn.Module):
    """CSI decoder module for VAE."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_dim: int,
        categorical_dim: int,
        out_filter: int,
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        flat_dim = input_shape[0] * input_shape[1] * input_shape[2]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim * categorical_dim, flat_dim),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(input_shape[2], 32, kernel_size=(2, 4), stride=(2, 4), padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 8), stride=(5, 8), padding=2, output_padding=(4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 8), stride=(5, 8), padding=2, output_padding=(4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_filter, kernel_size=out_filter),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        x = self.fc(z)
        x = x.view(
            z.size(0),
            self.input_shape[2],
            self.input_shape[0],
            self.input_shape[1],
        )
        return self.deconv(x)
