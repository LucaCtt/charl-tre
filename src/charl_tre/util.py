import os
import random
from pathlib import Path

import optuna
import torch
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


def make_dl(ds: Dataset, batch_size: int, shuffle: bool, n_workers: int, seed: int) -> DataLoader:
    """Create a DataLoader with common settings.

    Handles distributed training by using a DistributedSampler when torch.distributed is initialized.

    Arguments:
        ds: Dataset to load data from.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the data at the beginning of each epoch.
        n_workers: Number of subprocesses to use for data loading.
        seed: Random seed for reproducibility of shuffling.

    Returns:
        A DataLoader instance for the given dataset and settings.

    """
    is_ddp_spawned = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    n_workers = 0 if is_ddp_spawned else n_workers

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if not is_ddp_spawned else False,
        num_workers=n_workers,
        persistent_workers=n_workers > 0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
        sampler=DistributedSampler(ds, shuffle=shuffle, seed=seed) if is_ddp_spawned else None,
    )


def make_study(study_name: str, storage_dir: str | None = None, seed: int | None = None) -> optuna.Study:
    """Create (or load) an Optuna study backed by a SQLite database.

    Arguments:
        study_name (str): The name of the study to create or load.
        storage_dir (str | None): The directory to use for storage.
            If None, the storage will be in-memory and not persisted to disk.
        seed (int | None): The seed to use for the random number generator.

    Returns:
        An Optuna Study object.

    """
    if storage_dir:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        journal_path = f"{storage_dir}/{study_name}.sqlite"
    else:
        journal_path = ":memory:"

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{journal_path}",
        heartbeat_interval=60,
        grace_period=120,
        heartbeat_stale_trial_callback=optuna.storages.RetryHeartbeatStaleTrialCallback(max_retry=3),
    )

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        direction="maximize",
        load_if_exists=True,
    )


def read_study(launch_dir: str) -> optuna.Study:
    """Read an Optuna study from a SQLite file in the specified launch directory.

    Arguments:
        launch_dir (Path): The directory where the Optuna study SQLite file is located.

    Returns:
        optuna.Study: An Optuna Study object loaded from the SQLite file in the launch directory.

    """
    study_files = sorted([f.name for f in Path(launch_dir).iterdir() if f.is_file() and f.suffix == ".sqlite"])
    if len(study_files) == 0:
        msg = f"No study files found in {launch_dir}."
        raise FileNotFoundError(msg)

    study = study_files[-1]
    return optuna.load_study(study_name=study.split(".")[0], storage=f"sqlite:///{Path(launch_dir) / study}")
