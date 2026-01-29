import logging
import os
from pathlib import Path

import torch
from pythonjsonlogger.json import JsonFormatter
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from torch.distributed import destroy_process_group, init_process_group
from torch.multiprocessing.spawn import spawn
from torch.utils.data import DataLoader

from csi_vae_gumbel.dataset import get_splits
from csi_vae_gumbel.models import Classifier, MultiViewCategoricalVAE
from csi_vae_gumbel.settings import Settings
from csi_vae_gumbel.train import CheckpointManager, ClassifierTrainer, EarlyStopping, VAETrainer

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


def _train_vae(rank: int, train_dataloader: DataLoader) -> MultiViewCategoricalVAE:
    vae = MultiViewCategoricalVAE(
        window_size=settings.window_size,
        n_antennas=settings.n_antennas,
        n_categories=settings.n_activities,
        categorical_dim=settings.categorical_dim,
        hidden_latent_dim=settings.hidden_latent_dim,
    )

    optimizer = torch.optim.Adam(vae.parameters(), lr=settings.learning_rate)
    early_stopping = EarlyStopping(patience=settings.patience)
    checkpoint_manager = CheckpointManager(Path(settings.checkpoint_dir))

    with Progress(
        BarColumn(),
        TimeRemainingColumn(compact=True),
        TextColumn("Epoch: {task.fields[epoch]}/{task.fields[total_epochs]}"),
        TextColumn("Batch: {task.fields[batch]}/{task.total}"),
        TextColumn("Loss: {task.fields[loss]:.4f}  Recon: {task.fields[recon]:.4f}  KL: {task.fields[kl]:.4f}"),
        disable=rank != 0,
        console=console,
    ) as progress:
        epoch_task = progress.add_task(
            "Training",
            total=len(train_dataloader),
            total_epochs=settings.n_epochs,
            epoch=1,
            batch=0,
            loss=0.0,
            recon=0.0,
            kl=0.0,
        )

        def batch_callback(epoch: int, epoch_loss: float, epoch_recon: float, epoch_kl: float) -> None:
            logger.debug(
                "VAE epoch progress",
                extra={
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "recon_loss": epoch_recon,
                    "kl_loss": epoch_kl,
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
            if progress.finished and epoch + 1 < settings.n_epochs:
                progress.reset(epoch_task)

        vae_trainer = VAETrainer(
            vae,
            train_dataloader,
            optimizer,
            early_stopping,
            checkpoint_manager,
            rank,
            batch_callback=batch_callback,
        )
        vae_trainer.train(settings.n_epochs)

    return vae


def _train_classifier(rank: int, train_dataloader: DataLoader, vae: MultiViewCategoricalVAE) -> Classifier:
    classifier = Classifier(
        input_dim=settings.categorical_dim * settings.n_activities,
        output_dim=settings.n_activities,
        hidden_dim=128,
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=settings.learning_rate)

    with Progress(
        BarColumn(),
        TimeRemainingColumn(compact=True),
        TextColumn("Epoch: {task.fields[epoch]}/{task.fields[total_epochs]}"),
        TextColumn("Batch: {task.fields[batch]}/{task.total}"),
        TextColumn("Loss: {task.fields[loss]:.4f}  Accuracy: {task.fields[accuracy]:.4f}"),
        disable=rank != 0,
        console=console,
    ) as progress:
        epoch_task = progress.add_task(
            "Training Classifier",
            total=len(train_dataloader),
            total_epochs=settings.n_epochs,
            epoch=1,
            batch=0,
            loss=0.0,
            accuracy=0.0,
        )

        def batch_callback(epoch: int, epoch_loss: float, epoch_accuracy: float) -> None:
            logger.debug(
                "Classifier epoch progress",
                extra={
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "accuracy": epoch_accuracy,
                },
            )
            progress.update(
                epoch_task,
                advance=1,
                epoch=epoch + 1,
                batch=progress.tasks[0].completed + 1,
                loss=epoch_loss,
                accuracy=epoch_accuracy,
            )
            if progress.finished and epoch + 1 < settings.n_epochs:
                progress.reset(epoch_task)

        class_trainer = ClassifierTrainer(
            classifier,
            train_dataloader,
            vae,
            optimizer,
            batch_callback=batch_callback,
            gpu_id=rank,
        )
        class_trainer.train(settings.n_epochs)

    return classifier


def train(rank: int, world_size: int) -> None:
    """Train the VAE model using Distributed Data Parallel (DDP).

    Arguments:
        rank: Unique identifier of the process.
        world_size: Total number of processes.

    """
    _ddp_setup(rank, world_size)

    with Progress(
        SpinnerColumn(),
        TextColumn("Loading CSI data..."),
        console=console,
        disable=rank != 0,
    ) as progress:
        progress.add_task("load_data", total=None)

        train_dataloader, test_dataloader = get_splits(
            dataset_path=Path(settings.dataset_path),
            batch_size=settings.batch_size // world_size,
            window_size=settings.window_size,
            overlap_size=settings.overlap_size,
            n_activities=settings.n_activities,
            n_samples=settings.n_samples,
            n_antennas=settings.n_antennas,
        )

    vae = _train_vae(rank, train_dataloader)
    classifier = _train_classifier(rank, train_dataloader, vae)

    accuracy = 0.0

    for x, y in test_dataloader:
        _, logits_vae = vae(x.to(rank))
        logits_vae = logits_vae.view(logits_vae.size(0), -1)
        logits = classifier(logits_vae)
        preds = torch.argmax(logits, dim=1)
        accuracy += (preds == y.to(rank)).float().mean().item()

    accuracy /= len(test_dataloader)

    if rank == 0:
        logger.info(
            "Classification accuracy",
            extra={
                "accuracy": accuracy,
            },
        )

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
