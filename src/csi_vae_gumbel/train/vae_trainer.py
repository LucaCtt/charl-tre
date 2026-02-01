from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.loss import KLWeightAnnealer, vae_loss
from csi_vae_gumbel.loss.capacity_annealer import CapacityAnnealer
from csi_vae_gumbel.loss.entropy_annealer import EntropyAnnealer
from csi_vae_gumbel.loss.gumbel_annealer import GumbelTemperatureAnnealer
from csi_vae_gumbel.train.async_callback_worker import AsyncCallbackWorker
from csi_vae_gumbel.train.checkpoints import CheckpointManager


class VAETrainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        checkpoint_manager: CheckpointManager,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        kl_weight_annealer: KLWeightAnnealer,
        temperature_annealer: GumbelTemperatureAnnealer,
        entropy_annealer: EntropyAnnealer,
        capacity_annealer: CapacityAnnealer,
        loss_type: Literal["bce", "mse"],
        gpu_id: int,
        batch_callback: Callable | None = None,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for training.
            checkpoint_manager: CheckpointManager to save model checkpoints.
            lr_scheduler: Learning rate scheduler.
            kl_weight_annealer: Scheduler for KL divergence weight.
            temperature_annealer: Scheduler for Gumbel temperature.
            entropy_annealer: Scheduler for entropy weight.
            capacity_annealer: Scheduler for KL capacity.
            loss_type: Type of reconstruction loss ("bce" or "mse").
            gpu_id: GPU ID for Distributed Data Parallel.
            batch_callback: Optional callback function called at the end of each batch.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__checkpoint_manager = checkpoint_manager
        self.__gpu_id = gpu_id
        self.__batch_callback = batch_callback
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.__kl_annealer = kl_weight_annealer
        self.__temperature_annealer = temperature_annealer
        self.__entropy_annealer = entropy_annealer
        self.__capacity_annealer = capacity_annealer
        self.__loss_type: Literal["bce", "mse"] = loss_type

        self.__callback_worker = AsyncCallbackWorker()

    def __run_batch(
        self,
        x_true: torch.Tensor,
        tau: float,
        kl_weight: float,
        entropy_weight: float,
        entropy_mode: Literal["none", "penalty", "bonus"],
        capacity: float,
    ) -> tuple[float, float, float]:
        self.__optimizer.zero_grad()

        x_recon, _, logits = self.__model(x_true, tau)

        loss, recon_loss, kl_loss, _ = vae_loss(
            x_recon,
            x_true,
            logits,
            kl_weight=kl_weight,
            entropy_mode=entropy_mode,
            entropy_weight=entropy_weight,
            capacity=capacity,
            loss_type=self.__loss_type,
        )

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

        tau = self.__temperature_annealer.step(epoch)
        kl_weight = self.__kl_annealer.step(epoch)
        entropy_weight, entropy_mode = self.__entropy_annealer.step(epoch)
        capacity = self.__capacity_annealer.step(epoch)

        for i, (x_true, _) in enumerate(self.__dataloader):
            loss, recon_loss, kl_loss = self.__run_batch(
                x_true.to(self.__gpu_id),
                tau,
                kl_weight,
                entropy_weight,
                entropy_mode,
                capacity,
            )

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

        # This has to be called after each epoch
        self.__lr_scheduler.step(epoch_loss)

        return epoch_loss, epoch_recon_loss, epoch_kl_loss

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Will resume from the latest checkpoint if available.

        Arguments:
            epochs: Number of epochs to train.

        Returns:
            Tuple containing average total loss, reconstruction loss, and KL divergence loss over all epochs.

        """
        self.__model.train()

        self.__callback_worker.start()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        for epoch in range(epochs):
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

        total_loss /= epochs
        total_recon_loss /= epochs
        total_kl_loss /= epochs

        self.__callback_worker.stop()
        return total_loss, total_recon_loss, total_kl_loss
