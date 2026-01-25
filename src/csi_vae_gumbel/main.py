import logging
import os
import sys
from pathlib import Path

import torch
from pythonjsonlogger.json import JsonFormatter
from torch.distributed import destroy_process_group, init_process_group
from torch.multiprocessing.spawn import spawn

from csi_vae_gumbel.dataset import build_dataloader
from csi_vae_gumbel.model.vae import VAE
from csi_vae_gumbel.settings import Settings
from csi_vae_gumbel.train.checkpoints import CheckpointManager
from csi_vae_gumbel.train.early_stopping import EarlyStopping
from csi_vae_gumbel.train.trainer import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = JsonFormatter()
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)


def _ddp_setup(rank: int, world_size: int) -> None:
    """Initialize the distributed environment.

    Arguments:
        rank: Unique identifier of each process
        world_size: Total number of processes

    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    acc = torch.accelerator.current_accelerator()
    if acc is None:
        msg = "No accelerator found for DDP setup."
        raise RuntimeError(msg)

    backend = torch.distributed.get_default_backend_for_device(acc)

    init_process_group(backend=backend, rank=rank, world_size=world_size)


def train(rank: int, world_size: int) -> None:
    """Train the VAE model using Distributed Data Parallel (DDP).

    Arguments:
        rank: Unique identifier of the process.
        world_size: Total number of processes.

    """
    _ddp_setup(rank, world_size)
    settings = Settings()

    dataloader = build_dataloader(
        dataset_path=Path(settings.dataset_path),
        batch_size=settings.batch_size,
        window_size=settings.window_size,
        n_activities=settings.n_activities,
        n_samples=settings.n_samples,
        n_antennas=settings.n_antennas,
        antenna=settings.antenna,
    )
    model = VAE(latent_dim=settings.latent_dim, categorical_dim=settings.categorical_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    early_stopping = EarlyStopping(patience=settings.patience)
    checkpoint_manager = CheckpointManager(Path(settings.checkpoint_dir))

    trainer = Trainer(model, dataloader, optimizer, early_stopping, checkpoint_manager, rank)
    trainer.train(settings.n_epochs)

    destroy_process_group()


def main() -> None:
    """Spawn multiple processes for distributed training."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    spawn(train, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    main()
