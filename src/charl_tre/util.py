import json
import os
import random
from pathlib import Path

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def init_rng(seed: int) -> None:
    """Initialize random seeds for reproducibility.

    Arguments:
        seed (int): The random seed to use for all random number generators.

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def setup_ddp(rank: int, world_size: int) -> None:
    """Initialize the distributed environment. Must be called by every distributed process.

    Arguments:
        rank: Unique identifier of each distributed process
        world_size: Total number of distributed processes

    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    acc = torch.accelerator.current_accelerator()
    if acc is None:
        msg = "No accelerator found for DDP setup."
        raise RuntimeError(msg)
    backend = torch.distributed.get_default_backend_for_device(acc)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, device_id=rank)


def make_dl(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    """Create a DataLoader with common settings.

    Arguments:
        ds: Dataset to load data from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the beginning of each epoch.
        num_workers: Number of subprocesses to use for data loading.
        seed: Random seed for reproducibility of shuffling.

    Returns:
        A DataLoader instance for the given dataset and settings.

    """
    is_ddp_spawned = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    num_workers = 0 if is_ddp_spawned else num_workers

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if not is_ddp_spawned else False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
        sampler=DistributedSampler(ds, shuffle=shuffle, seed=seed) if is_ddp_spawned else None,
    )
