import logging
import os
from pathlib import Path

import torch
from pythonjsonlogger.json import JsonFormatter
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.distributed import destroy_process_group, init_process_group
from torch.multiprocessing.spawn import spawn

from csi_vae_gumbel.dataset import build_dataloader
from csi_vae_gumbel.model.vae import VAE
from csi_vae_gumbel.settings import Settings
from csi_vae_gumbel.train.checkpoints import CheckpointManager
from csi_vae_gumbel.train.early_stopping import EarlyStopping
from csi_vae_gumbel.train.trainer import Trainer

settings = Settings()

level = logging.DEBUG if settings.debug else logging.INFO
console = Console()
handler = RichHandler(level=level, show_path=False, console=console)
formatter = JsonFormatter()
handler.setFormatter(formatter)

logging.basicConfig(level=level, handlers=[handler])

logger = logging.getLogger("rich")


def _ddp_setup(rank: int, world_size: int) -> None:
    """Initialize the distributed environment.

    Arguments:
        rank: Unique identifier of each process
        world_size: Total number of processes

    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    acc = torch.accelerator.current_accelerator()
    if acc is None:
        msg = "No accelerator found for DDP setup."
        raise RuntimeError(msg)

    backend = torch.distributed.get_default_backend_for_device(acc)

    init_process_group(backend=backend, rank=rank, world_size=world_size)
    logger.info("DDP initialized", extra={"gpu_id": rank, "n_gpus": world_size, "backend": backend})


def train(rank: int, world_size: int) -> None:
    """Train the VAE model using Distributed Data Parallel (DDP).

    Arguments:
        rank: Unique identifier of the process.
        world_size: Total number of processes.

    """
    _ddp_setup(rank, world_size)

    if rank == 0:
        logger.info("Starting training", extra=settings.model_dump())

    try:
        dataloader = build_dataloader(
            dataset_path=Path(settings.dataset_path),
            batch_size=settings.batch_size // world_size,
            window_size=settings.window_size,
            n_activities=settings.n_activities,
            n_samples=settings.n_samples,
            n_antennas=settings.n_antennas,
            antenna=settings.antenna,
        )
        logger.info("DataLoader built", extra={"gpu_id": rank})

        model = VAE(latent_dim=settings.latent_dim, categorical_dim=settings.categorical_dim)
        logger.info("Model initialized", extra={"gpu_id": rank})

        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        early_stopping = EarlyStopping(patience=settings.patience)
        checkpoint_manager = CheckpointManager(Path(settings.checkpoint_dir))

        bar_column = BarColumn()
        epoch_column = TextColumn("Epoch: {task.fields[epoch]}/{task.fields[total_epochs]}")
        batch_column = TextColumn("Batch: {task.completed}/{task.total}")
        loss_column = TextColumn("Loss: {task.fields[loss]:.4f}")
        recon_column = TextColumn("Recon: {task.fields[recon]:.4f}")
        kl_column = TextColumn("KL: {task.fields[kl]:.4f}")

        with Progress(
            bar_column,
            TimeRemainingColumn(compact=True),
            epoch_column,
            batch_column,
            loss_column,
            recon_column,
            kl_column,
            disable=rank != 0,
            console=console,
        ) as progress:
            epoch_task = progress.add_task(
                "Training...",
                total=len(dataloader),
                epoch=1,
                total_epochs=settings.n_epochs,
                batch=1,
                loss=0.0,
                recon=0.0,
                kl=0.0,
            )

            def epoch_callback(epoch: int, epoch_loss: float, epoch_recon: float, epoch_kl: float) -> None:
                logger.info(
                    "Epoch completed",
                    extra={
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "recon_loss": epoch_recon,
                        "kl_loss": epoch_kl,
                        "gpu_id": rank,
                    },
                )

            def batch_callback(epoch: int, epoch_loss: float, epoch_recon: float, epoch_kl: float) -> None:
                logger.debug(
                    "Epoch progress",
                    extra={
                        "epoch": epoch + 1,
                        "loss": epoch_loss,
                        "recon_loss": epoch_recon,
                        "kl_loss": epoch_kl,
                        "gpu_id": rank,
                    },
                )
                progress.update(
                    epoch_task,
                    advance=1,
                    epoch=epoch + 1,
                    batch=progress.tasks[0].completed + 1,
                    loss=epoch_loss,
                    recon=epoch_recon,
                    kl=epoch_kl,
                )
                if progress.finished:
                    progress.reset(epoch_task)

            trainer = Trainer(
                model,
                dataloader,
                optimizer,
                early_stopping,
                checkpoint_manager,
                rank,
                batch_callback=batch_callback,
                epoch_callback=epoch_callback,
            )
            trainer.train(settings.n_epochs)
    finally:
        destroy_process_group()


def main() -> None:
    """Spawn multiple propocesses for distributed training."""
    if settings.debug:
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    spawn(train, args=(world_size,), nprocs=world_size)


if __name__ == "__main__":
    main()
