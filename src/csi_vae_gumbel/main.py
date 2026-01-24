"""Main module to train the VAE model using DDP."""

import os

import torch
from torch.distributed import destroy_process_group, init_process_group

from csi_vae_gumbel.dataset import build_dataloader
from csi_vae_gumbel.model.vae import VAE
from csi_vae_gumbel.settings import Settings
from csi_vae_gumbel.train.checkpoints import CheckpointManager
from csi_vae_gumbel.train.early_stopping import EarlyStopping
from csi_vae_gumbel.train.trainer import Trainer


def ddp_setup(rank: int, world_size: int) -> None:
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


def main(rank: int, world_size: int) -> None:
    """Train the VAE model using Distributed Data Parallel (DDP).

    Arguments:
        rank: Unique identifier of the process.
        world_size: Total number of processes.

    """
    ddp_setup(rank, world_size)
    settings = Settings()

    dataloader = build_dataloader(
        dataset_path=settings.dataset_path,
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
    checkpoint_manager = CheckpointManager(settings.checkpoint_dir)
    trainer = Trainer(model, dataloader, optimizer, early_stopping, checkpoint_manager, rank)
    trainer.train(settings.n_epochs)

    destroy_process_group()
