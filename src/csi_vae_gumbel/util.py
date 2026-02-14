import torch


def split_test_window(x: torch.Tensor, n_windows: int, overlap_size: int) -> torch.Tensor:
    """Split every x along the window size dimension into separate samples.

    Args:
        x: (batch_size, n_antennas, test_window_size, n_subcarriers) input tensor
        n_windows: how many windows to split the test window into
        overlap_size: how many frames to overlap between windows

    Returns:
        (batch_size * n_windows, n_antennas, train_window_size, n_subcarriers) output tensor,
        where train_window_size = test_window_size // n_windows

    """
    test_window_size = x.shape[2]

    if test_window_size % n_windows != 0:
        msg = f"test_window_size {x.shape[2]} is not divisible by n_windows {n_windows}"
        raise ValueError(msg)

    out_window_size = x.shape[2] // n_windows

    if overlap_size >= out_window_size:
        msg = f"overlap_size {overlap_size} must be less than out_window_size {out_window_size}"
        raise ValueError(msg)

    step_size = out_window_size - overlap_size
    out = []
    for i in range(n_windows):
        start = i * step_size
        end = start + out_window_size
        out.append(x[:, :, start:end, :])

    return torch.cat(out, dim=0)
