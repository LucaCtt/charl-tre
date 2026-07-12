import torch
from torch import nn

from charl_tre.models import util

_N_DIMS_MULTIPLE_ANTENNAS: int = 4


def _split_test_window(x: torch.Tensor, sample_window_size: int, overlap_size: int) -> torch.Tensor:
    """Split every x along the window size dimension into separate samples.

    Args:
        x (torch.Tensor): (batch_size, n_antennas, in_window_size, n_subcarriers) input tensor.
        sample_window_size (int): The size of the windows to split the input into.
        overlap_size (int): how many frames to overlap between windows.

    Returns:
        torch.Tensor: (batch_size * n_windows, n_antennas, sample_window_size, n_subcarriers) output tensor,
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


class Classifier(nn.Module):
    """Classifier for delayed fusion of Dirichlet latents from multiple antennas."""

    def __init__(
        self,
        antennas: list[nn.Module],
        n_components: int,
        n_activities: int,
        sample_window_size: int,
        overlap_size: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        """Initialize the classifier for delayed fusion of Dirichlet latents.

        Arguments:
            antennas (list[nn.Module]): List of trained modules for each antenna.
            n_components (int): Number of components in the Dirichlet distribution.
            n_activities (int): Number of activity classes.
            sample_window_size (int): Size of the sample window.
            overlap_size (int): Size of the overlap between windows.
            n_layers (int): Number of layers in the fully connected network.
            dropout (float): Dropout probability.

        """
        super().__init__()

        self._sample_window_size = sample_window_size
        self._overlap_size = overlap_size

        self._antennas = nn.ModuleList(antennas)
        for param in self._antennas.parameters():
            param.requires_grad = False

        self._fc = util.build_fc(n_components * len(antennas), n_activities, n_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the delayed fusion module.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, n_antennas, window_size, n_subcarriers)
                if num_antennas > 1, or (batch_size, window_size, n_subcarriers) if num_antennas=1.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_activities).

        """
        x_r = _split_test_window(x, self._sample_window_size, self._overlap_size)

        batch_size = x.shape[0]
        n_windows = x_r.shape[0] // batch_size

        outs = []
        for i, antenna in enumerate(self._antennas):
            # Early fusion modules consume all antennas at once, while delayed-fusion
            # antenna modules consume a single antenna slice.
            if x_r.ndim == _N_DIMS_MULTIPLE_ANTENNAS and not hasattr(antenna, "_n_antennas"):
                antenna_in = x_r[:, i]
            else:
                antenna_in = x_r

            model_output = antenna(antenna_in)
            alpha = model_output[2]

            alpha = alpha.reshape(batch_size, n_windows, -1).mean(dim=1)
            outs.append(alpha)

        z = torch.cat(outs, dim=1)
        return self._fc(z)
