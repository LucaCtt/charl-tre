import json
import os
from pathlib import Path

import torch
from torch import distributed as dist


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


def get_best_model_path(study_path: Path) -> Path:
    """Get the path to the best trial model from the Optuna study results.

    Arguments:
        study_path: Path to the Optuna study results directory.

    Returns:
        The path to the best trial model.

    """
    with (study_path / "study_results.json").open("r") as f:
        study_info = json.load(f)
        best_trial_number = study_info["best_trial"]

    return Path(study_path) / f"trial_{best_trial_number}"


def get_vae_params(model_path: Path) -> dict:
    """Get the VAE parameters from the given model path.

    Note: this does not return a vae.Parameters object to avoid a circular dependency.

    Arguments:
        model_path: Path to the model directory containing the results.json file.

    Returns:
        The VAE parameters as a dictionary.

    """
    with (model_path / "results.json").open("r") as f:
        info = json.load(f)

    return {
        "final_cap": info["final_cap"],
        "start_gumbel_temp": info["start_gumbel_temp"],
        "final_kl_weight": info["final_kl_weight"],
        "latent_dim": info["latent_dim"],
    }


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize the distributed environment. Must be called by every distributed process.

    Arguments:
        rank: Unique identifier of each distributed process
        world_size: Total number of distributed processes

    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12356"

    acc = torch.accelerator.current_accelerator()
    if acc is None:
        msg = "No accelerator found for DDP setup."
        raise RuntimeError(msg)
    backend = torch.distributed.get_default_backend_for_device(acc)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=rank)
