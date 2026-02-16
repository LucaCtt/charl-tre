import torch


def split_test_window(x: torch.Tensor, sample_window_size: int, overlap_size: int) -> torch.Tensor:
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
        msg = "sample_window_size must be less than or equal to the window size dimension of x."
        raise ValueError(msg)

    in_window_size = x.shape[2]
    n_windows = (in_window_size - overlap_size) // (sample_window_size - overlap_size)

    if (in_window_size - overlap_size) % (sample_window_size - overlap_size) != 0:
        msg = (
            "The given sample window size and overlap size do not allow "
            "for an integer number of windows to be created from the in_window_size."
        )

        raise ValueError(msg)

    step_size = sample_window_size - overlap_size

    out = []
    for i in range(n_windows):
        start = i * step_size
        end = start + sample_window_size
        out.append(x[:, :, start:end, :])

    return torch.cat(out, dim=0)
