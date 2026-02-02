from typing import Literal

import optuna
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from csi_vae_gumbel.train.annealers import (
    CapacityAnnealer,
    EntropyAnnealer,
    GumbelTemperatureAnnealer,
    KLWeightAnnealer,
)
from csi_vae_gumbel.train.early_stopping import EarlyStopping
from csi_vae_gumbel.train.vae_loss import vae_loss
from csi_vae_gumbel.train.vae_parameters import VAEParameters


class VAETrainer:
    """Trainer class for VAE model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        parameters: VAEParameters,
        gpu_id: int,
        trial: optuna.integration.TorchDistributedTrial,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: VAE model to be trained.
            dataloader: DataLoader for training data.
            parameters: VAE training parameters.
            gpu_id: GPU ID for Distributed Data Parallel.
            trial: Optuna trial for hyperparameter optimization (optional).
            callback: Optional callback function to be called after each epoch.

        """
        self.__model = DistributedDataParallel(model.to(gpu_id), device_ids=[gpu_id])
        self.__dataloader = dataloader
        self.__optimizer = torch.optim.Adam(
            model.parameters(),
            lr=parameters.start_lr,
        )
        self.__lr_annealer = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.__optimizer,
            mode="min",
        )
        self.__capacity_annealer = CapacityAnnealer(
            final_capacity=parameters.final_cap,
        )
        self.__entropy_annealer = EntropyAnnealer(
            final_weight=parameters.final_entr_weight,
        )
        self.__temperature_annealer = GumbelTemperatureAnnealer(
            start_tau=parameters.gumbel_temp,
            min_tau=parameters.gumbel_temp / 10,
        )
        self.__kl_weight_annealer = KLWeightAnnealer(
            max_weight=parameters.final_kl_weight,
        )
        self.__loss_type: Literal["bce", "mse"] = parameters.loss_type
        self.__early_stopping = EarlyStopping()

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
    ) -> tuple[float, float, float, float]:
        self.__optimizer.zero_grad()

        x_recon, _, logits = self.__model(x_true, tau)

        loss, recon_loss, kl_loss, entropy_loss = vae_loss(
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

        return loss.item(), recon_loss.item(), kl_loss.item(), entropy_loss.item()

    def __run_epoch(self, epoch: int) -> tuple[float, float, float, float]:
        # Set the epoch for shuffling if using DistributedSampler
        if isinstance(self.__dataloader.sampler, DistributedSampler):
            self.__dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_entropy_loss = 0.0

        tau = self.__temperature_annealer.step(epoch)
        kl_weight = self.__kl_weight_annealer.step(epoch)
        entropy_weight, entropy_mode = self.__entropy_annealer.step(epoch)
        capacity = self.__capacity_annealer.step(epoch)

        for x_true, _ in self.__dataloader:
            loss, recon_loss, kl_loss, entropy_loss = self.__run_batch(
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
            epoch_entropy_loss += entropy_loss

        epoch_loss /= len(self.__dataloader)
        epoch_recon_loss /= len(self.__dataloader)
        epoch_kl_loss /= len(self.__dataloader)
        epoch_entropy_loss /= len(self.__dataloader)

        # This has to be called after each epoch
        self.__lr_annealer.step(epoch_loss)

        self.__trial.report(epoch_loss, step=epoch)

        if self.__trial.should_prune():
            raise optuna.TrialPruned

        return epoch_loss, epoch_recon_loss, epoch_kl_loss, epoch_entropy_loss

    def train(self, epochs: int) -> tuple[float, float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Number of epochs to train.
            max_epochs_zero_kl: Maximum number of consecutive epochs with near-zero KL divergence before pruning.
            eps: Threshold to consider KL divergence as near-zero.

        Returns:
            Tuple containing average total loss, reconstruction loss,
            KL divergence loss, and entropy loss over all epochs.

        """
        self.__model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_entropy_loss = 0.0

        for epoch in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss, epoch_entropy_loss = self.__run_epoch(epoch)

            total_loss += epoch_loss
            total_recon_loss += epoch_recon_loss
            total_kl_loss += epoch_kl_loss
            total_entropy_loss += epoch_entropy_loss

            should_stop = self.__early_stopping.step(epoch_loss, epoch_kl_loss)
            if should_stop:
                msg = f"Early stopping triggered at epoch {epoch}."
                raise optuna.TrialPruned(msg)

        total_loss /= epochs
        total_recon_loss /= epochs
        total_kl_loss /= epochs
        total_entropy_loss /= epochs

        return total_loss, total_recon_loss, total_kl_loss, total_entropy_loss
