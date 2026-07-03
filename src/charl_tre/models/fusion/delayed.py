import math

import torch
from torch import nn

from charl_tre.models.vae.dirichlet import SingleAntenna

_N_DIMS_MULTIPLE_ANTENNAS: int = 4


def _next_multiple_of_8(n: int) -> int:
    """Round n up to the next multiple of 8."""
    return math.ceil(n / 8) * 8


def _build_fc(in_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """Build FC block with n_layers, keeping hidden dims as multiples of 8."""
    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    # Geometrically interpolate hidden dims, only internal ones must be multiples of 8
    dims = (
        [in_dim]
        + [_next_multiple_of_8(int(in_dim * ((out_dim / in_dim) ** (i / (n_layers - 1))))) for i in range(1, n_layers)]
        + [out_dim]
    )

    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < n_layers - 1:
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.GELU())

    return nn.Sequential(*layers)

def _split_test_window(x: torch.Tensor, sample_window_size: int, overlap_size: int) -> torch.Tensor:
    """Split every x along the window size dimension into separate samples.

    Args:
        x: (batch_size, n_antennas, in_window_size, n_subcarriers) input tensor.
        sample_window_size: The size of the windows to split the input into.
        overlap_size: how many frames to overlap between windows.

    Returns:
        (batch_size * n_windows, n_antennas, sample_window_size, n_subcarriers) output tensor,
        where n_windows is the number of windows that can be created from the in_window_size
        given the sample window size and overlap size.

    """
    if sample_window_size < overlap_size:
        msg = "sample_window_size must be greater than or equal to overlap_size."
        raise ValueError(msg)

    if sample_window_size > x.shape[2]:
        msg = "sample_window_size must be less than or equal to the window size of x."
        raise ValueError(msg)

    if overlap_size < 0:
        msg = "overlap_size must be non-negative."
        raise ValueError(msg)

    batch_size, n_antennas, in_window_size, n_subcarriers = x.shape

    step = sample_window_size - overlap_size

    # Shape will be (batch_size, n_antennas, n_windows, sample_window_size, n_subcarriers)
    x_unfold = x.unfold(dimension=2, size=sample_window_size, step=step)
    n_windows = x_unfold.shape[2]

    expected_window_size = step * (n_windows - 1) + sample_window_size
    if expected_window_size != in_window_size:
        msg = "Window configuration does not exactly tile the time dimension."
        raise ValueError(msg)

    # Reorder so windows are grouped per sample
    # Shape will be (batch_size, n_windows, n_antennas, sample_window_size, n_subcarriers)
    x_unfold = x_unfold.permute(0, 2, 1, 3, 4).contiguous()

    # Merge batch and window dimensions
    # Shape will be (batch_size * n_windows, n_antennas, sample_window_size, n_subcarriers)
    return x_unfold.view(batch_size * n_windows, n_antennas, sample_window_size, n_subcarriers)


class Delayed(nn.Module):
    """Delayed fusion module for multi-antenna CSI data."""

    def __init__(
        self,
        antennas: list[SingleAntenna],
        n_components: int,
        n_activities: int,
        sample_window_size: int,
        overlap_size: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the delayed fusion module for Dirichlet latents."""
        super().__init__()

        self._sample_window_size = sample_window_size
        self._overlap_size = overlap_size

        self._antennas = nn.ModuleList(antennas)
        for param in self._antennas.parameters():
            param.requires_grad = False

        self._fc = _build_fc(n_components * len(antennas), n_activities, n_layers, dropout)

    @property
    def antennas(self) -> list[SingleAntenna]:
        """Return the list of SingleAntenna modules."""
        return list(self._antennas)  # pyright: ignore[reportReturnType]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the delayed fusion module.

        Arguments:
            x: Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers) if num_antennas > 1,
                or (batch_size, window_size, n_subcarriers) if num_antennas=1.

        Returns:
            Output tensor of shape (batch_size, n_activities).

        """
        x_r = _split_test_window(x, self._sample_window_size, self._overlap_size)

        batch_size = x.shape[0]
        n_windows = x_r.shape[0] // batch_size

        outs = []
        for i, antenna in enumerate(self._antennas):
            _, alpha = antenna(x_r[:, i] if x_r.ndim == _N_DIMS_MULTIPLE_ANTENNAS else x_r)
            alpha = alpha.reshape(batch_size, n_windows, -1).mean(dim=1)
            outs.append(alpha)

        z = torch.cat(outs, dim=1)
        return self._fc(z)
