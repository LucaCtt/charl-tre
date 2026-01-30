from collections.abc import Callable

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.loss import KLScheduler, vae_loss
from csi_vae_gumbel.models.gumbel_annealer import GumbelAnnealer
from csi_vae_gumbel.train.async_callback_worker import AsyncCallbackWorker
from csi_vae_gumbel.train.checkpoints import CheckpointManager


class VAETrainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_manager: CheckpointManager,
        gpu_id: int,
        batch_callback: Callable | None = None,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for training.
            checkpoint_manager: CheckpointManager to save model checkpoints.
            gpu_id: GPU ID for Distributed Data Parallel.
            batch_callback: Optional callback function called at the end of each batch.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__optimizer = optimizer
        self.__checkpoint_manager = checkpoint_manager
        self.__gpu_id = gpu_id
        self.__batch_callback = batch_callback
        self.__kl_scheduler = KLScheduler()
        self.__gumbel_annealer = GumbelAnnealer()

        self.__callback_worker = AsyncCallbackWorker()

    def __run_batch(self, epoch: int, x_true: torch.Tensor) -> tuple[float, float, float]:
        self.__optimizer.zero_grad()

        tau = self.__gumbel_annealer.step(epoch)
        x_recon, _, logits = self.__model(x_true, tau)

        kl_weight = self.__kl_scheduler.get_weight(epoch)
        loss, recon_loss, kl_loss, _ = vae_loss(x_recon, x_true, logits, kl_weight=kl_weight)

        loss.backward()
        self.__optimizer.step()

        return loss.item(), recon_loss.item(), kl_loss.item()

    def __run_epoch(self, epoch: int) -> tuple[float, float, float]:
        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0

        for i, (x_true, _) in enumerate(self.__dataloader):
            loss, recon_loss, kl_loss = self.__run_batch(epoch, x_true.to(self.__gpu_id))

            epoch_loss += loss
            epoch_recon_loss += recon_loss
            epoch_kl_loss += kl_loss

            if self.__gpu_id == 0 and self.__batch_callback is not None:
                done_batches = i + 1
                self.__callback_worker.submit(
                    self.__batch_callback,
                    epoch,
                    epoch_loss / done_batches,
                    epoch_recon_loss / done_batches,
                    epoch_kl_loss / done_batches,
                )

        epoch_loss /= len(self.__dataloader)
        epoch_recon_loss /= len(self.__dataloader)
        epoch_kl_loss /= len(self.__dataloader)

        return epoch_loss, epoch_recon_loss, epoch_kl_loss

    def train(self, max_epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Will resume from the latest checkpoint if available.

        Arguments:
            max_epochs: Number of epochs to train, including any previously completed epochs.

        Returns:
            Tuple containing average total loss, reconstruction loss, and KL divergence loss over all epochs.

        """
        self.__model.train()

        self.__callback_worker.start()

        latest_checkpoint = self.__checkpoint_manager.load_latest_checkpoint()
        if latest_checkpoint is not None:
            model_state, optimizer_state, start_epoch = latest_checkpoint
            self.__model.module.load_state_dict(model_state)
            self.__optimizer.load_state_dict(optimizer_state)
        else:
            start_epoch = 0

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        for epoch in range(start_epoch, max_epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self.__run_epoch(epoch)

            total_loss += epoch_loss
            total_recon_loss += epoch_recon_loss
            total_kl_loss += epoch_kl_loss

            if self.__gpu_id == 0:
                self.__checkpoint_manager.save_checkpoint(
                    self.__model.module.state_dict(),
                    self.__optimizer.state_dict(),
                    epoch + 1,
                )

        total_loss /= max_epochs - start_epoch
        total_recon_loss /= max_epochs - start_epoch
        total_kl_loss /= max_epochs - start_epoch

        self.__callback_worker.stop()
        return total_loss, total_recon_loss, total_kl_loss
