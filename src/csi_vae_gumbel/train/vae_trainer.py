from typing import Literal

import optuna
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.loss import KLWeightAnnealer, vae_loss
from csi_vae_gumbel.loss.capacity_annealer import CapacityAnnealer
from csi_vae_gumbel.loss.entropy_annealer import EntropyAnnealer
from csi_vae_gumbel.loss.gumbel_annealer import GumbelTemperatureAnnealer

_EPS = 1e-6
_MAX_ZERO_KL_EPOCHS = 3


class VAETrainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        kl_weight_annealer: KLWeightAnnealer,
        temperature_annealer: GumbelTemperatureAnnealer,
        entropy_annealer: EntropyAnnealer,
        capacity_annealer: CapacityAnnealer,
        loss_type: Literal["bce", "mse"],
        gpu_id: int,
        trial: optuna.integration.TorchDistributedTrial,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for training.
            lr_scheduler: Learning rate scheduler.
            kl_weight_annealer: Scheduler for KL divergence weight.
            temperature_annealer: Scheduler for Gumbel temperature.
            entropy_annealer: Scheduler for entropy weight.
            capacity_annealer: Scheduler for KL capacity.
            loss_type: Type of reconstruction loss ("bce" or "mse").
            gpu_id: GPU ID for Distributed Data Parallel.
            trial: Optuna trial for hyperparameter optimization (optional).
            callback: Optional callback function to be called after each epoch.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.__kl_annealer = kl_weight_annealer
        self.__temperature_annealer = temperature_annealer
        self.__entropy_annealer = entropy_annealer
        self.__capacity_annealer = capacity_annealer
        self.__loss_type: Literal["bce", "mse"] = loss_type

        self.__gpu_id = gpu_id
        self.__trial = trial

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

        for x_true, _ in self.__dataloader:
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

        epoch_loss /= len(self.__dataloader)
        epoch_recon_loss /= len(self.__dataloader)
        epoch_kl_loss /= len(self.__dataloader)

        # This has to be called after each epoch
        self.__lr_scheduler.step(epoch_loss)

        self.__trial.report(epoch_loss, step=epoch)

        if self.__trial.should_prune():
            raise optuna.TrialPruned

        return epoch_loss, epoch_recon_loss, epoch_kl_loss

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Number of epochs to train.

        Returns:
            Tuple containing average total loss, reconstruction loss, and KL divergence loss over all epochs.

        """
        self.__model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        zero_kl_epochs = 0

        for epoch in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self.__run_epoch(epoch)

            total_loss += epoch_loss
            total_recon_loss += epoch_recon_loss
            total_kl_loss += epoch_kl_loss

            if epoch_kl_loss < _EPS:
                zero_kl_epochs += 1
            else:
                zero_kl_epochs = 0

        if zero_kl_epochs >= _MAX_ZERO_KL_EPOCHS:
            msg = f"KL collapse detected for {zero_kl_epochs} consecutive epochs"
            raise optuna.TrialPruned(msg)

        total_loss /= epochs
        total_recon_loss /= epochs
        total_kl_loss /= epochs

        return total_loss, total_recon_loss, total_kl_loss
